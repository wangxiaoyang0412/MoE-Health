"""Model factory and registry."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

__all__ = ["MODEL_FACTORY", "ModelFactory", "register_model"]

MODEL_FACTORY: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Any:
    """Register an ``nn.Module`` class under ``name``."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        MODEL_FACTORY[name] = cls
        return cls

    return decorator


def ModelFactory(name: str) -> type[nn.Module]:  # noqa: N802 - factory naming convention
    if name not in MODEL_FACTORY:
        raise KeyError(f"unknown model '{name}'; registered: {sorted(MODEL_FACTORY)}")
    return MODEL_FACTORY[name]
