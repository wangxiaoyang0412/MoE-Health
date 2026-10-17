"""MoE-Health component tests on synthetic tensors."""

from __future__ import annotations

import numpy as np
import torch

from src.data_module.dataset import ffill_time, prepare_ehr
from src.data_module.ehr_stats import series_stats
from src.model_module.moe import ALL_COMBOS, MoELayer, balance_loss, combo_code
from src.model_module.moe_health import MoEHealth
from src.trainer_module.moe_procedure import pretrain_subset
from src.utils.config import ModelConfig

TS_DIM, CARDS, NOTE, CXR = 10, [3, 4, 2, 3, 2], 12, 16


def toy_batch(n: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(n, 3)
    mask[: n // 2, 2] = 0  # first half lacks CXR
    mask[::3, 1] = 0  # every third lacks notes
    return {
        "ehr_ts": torch.randn(n, 48, TS_DIM, generator=g),
        "static_num": torch.randn(n, 1, generator=g),
        "static_cat": torch.stack([torch.randint(0, c, (n,), generator=g) for c in CARDS], dim=1),
        "note": torch.randn(n, NOTE, generator=g),
        "cxr": torch.randn(n, CXR, generator=g),
        "mask": mask,
        "combo": (mask * torch.tensor([1, 2, 4])).sum(1).long(),
        "label": torch.randint(0, 2, (n,), generator=g).float(),
        "row": torch.arange(n),
    }


def toy_model(**kw: object) -> MoEHealth:
    cfg = ModelConfig(d_h=16, n_heads=4, lstm_hidden=8, static_emb_dim=4, expert_hidden=32, gate_hidden=16,
                      head_hidden=8, **kw)  # type: ignore[arg-type]
    return MoEHealth(cfg, TS_DIM, CARDS, NOTE, CXR, list(ALL_COMBOS))


def test_combo_order_matches_fig2() -> None:
    names = ["".join(L for L, b in zip("ETI", c, strict=True) if b) for c in ALL_COMBOS]
    assert names == ["E", "T", "I", "ET", "EI", "TI", "ETI"]
    assert [combo_code(c) for c in ALL_COMBOS] == [1, 2, 4, 3, 5, 6, 7]


def test_forward_modes_and_routing() -> None:
    model, batch = toy_model(), toy_batch()
    out = model(batch)
    assert out["logits"].shape == (8,)
    assert torch.allclose(out["gates"].sum(-1), torch.ones(8), atol=1e-5)
    assert out["topk_idx"].shape == (8, 2)
    assert torch.allclose(out["topk_w"].sum(-1), torch.ones(8), atol=1e-5)
    assert model(batch, mode="dense")["logits"].shape == (8,)
    assert model(batch, mode="expert", expert=3)["logits"].shape == (8,)


def test_absent_modalities_use_missing_embeddings() -> None:
    model, batch = toy_model(), toy_batch()
    x = model.tokens(batch, batch["mask"])
    assert torch.allclose(x[0, 2], model.absent[2])  # sample 0 lacks CXR
    assert torch.allclose(x[0, 1], model.absent[1])  # sample 0 lacks notes
    zero = toy_model(missing="zeros")
    tokens = zero.tokens(batch, batch["mask"])
    assert torch.count_nonzero(tokens[0, 2]) == 0
    assert torch.count_nonzero(zero.attention_input(tokens, batch["mask"])[0, 2]) == 0  # no type embedding


def test_top1_gate_receives_task_gradient() -> None:
    layer = MoELayer(12, 1, 7, 16, 8, top_k=1, gating="learned", dropout=0.0)
    y, aux = layer(torch.randn(32, 12))
    assert torch.allclose(aux["topk_w"], torch.ones(32, 1))
    torch.nn.functional.binary_cross_entropy_with_logits(y.squeeze(-1), torch.ones(32)).backward()
    assert layer.gate.net[0].weight.grad is not None and layer.gate.net[0].weight.grad.abs().sum() > 0


def test_logit_expert_variant_and_uniform_gating() -> None:
    batch = toy_batch()
    assert toy_model(expert_output="logit")(batch)["logits"].shape == (8,)
    out = toy_model(gating="uniform")(batch)
    assert torch.allclose(out["topk_w"], torch.full((8, 7), 1 / 7))


def test_balance_loss_gradient_and_uniform_zero() -> None:
    layer = MoELayer(12, 12, 7, 16, 8, top_k=2, gating="learned", dropout=0.0)
    _, aux = layer(torch.randn(64, 12))
    loss = balance_loss(aux["gates"], aux["topk_idx"], "cv")
    loss.backward()
    assert layer.gate.net[0].weight.grad is not None and float(loss) > 0
    gates = torch.full((7, 7), 1 / 7)
    idx = torch.arange(7).unsqueeze(1)
    assert float(balance_loss(gates, idx, "cv")) < 1e-6
    assert np.isclose(float(balance_loss(gates, idx, "switch")), 1.0)


def test_pretrain_subset_policies() -> None:
    codes = torch.tensor([1, 3, 5, 7])
    assert pretrain_subset(codes, (1, 0, 0), "superset").tolist() == [True] * 4
    assert pretrain_subset(codes, (1, 1, 0), "superset").tolist() == [False, True, False, True]
    assert pretrain_subset(codes, (1, 1, 0), "exact").tolist() == [False, True, False, False]
    assert pretrain_subset(codes, (0, 1, 1), "superset").tolist() == [False, False, False, True]


def test_ffill_and_prepare_ehr() -> None:
    x = np.array([[[np.nan], [1.0], [np.nan], [3.0]]], dtype=np.float32)
    assert np.allclose(ffill_time(x)[0, :, 0], [np.nan, 1.0, 1.0, 3.0], equal_nan=True)
    vals = np.array([[[np.nan], [1.0]], [[3.0], [5.0]]], dtype=np.float32)
    act = np.zeros((2, 2, 1), dtype=np.uint8)
    out = prepare_ehr(vals, act, np.array([0, 1]), 0.0, 1.0)
    assert out.shape == (2, 2, 3)
    assert out[0, 0, 0] == 0.0 and out[0, 0, 1] == 0.0  # unobserved -> train mean (0), mask 0
    assert np.isclose(out[:, :, 0][out[:, :, 1] > 0].mean(), 0.0, atol=1e-6)


def test_series_stats() -> None:
    z = np.array([[1.0, 0.0, 3.0, 2.0, 0.0]])
    obs = np.array([[True, False, True, True, False]])
    s = series_stats(z, obs)[0]
    # observed sequence [1, 3, 2]: max 3, min 1, mean 2, diffs [2, -1], last-first 1, one peak
    assert s[0] == 3 and s[1] == 1 and np.isclose(s[2], 2) and np.isclose(s[4], 0.5)
    assert np.isclose(s[6], 2) and np.isclose(s[8], 1) and s[9] == 1
    assert np.all(series_stats(np.zeros((1, 5)), np.zeros((1, 5), bool)) == 0)


def test_hours_since_last() -> None:
    from src.data_module.dataset import hours_since_last

    obs = np.array([[[False], [True], [False], [False]]])
    out = hours_since_last(obs)[0, :, 0] * 4
    assert np.allclose(out, [4.0, 0.0, 1.0, 2.0])  # never observed yet -> T (=4)
