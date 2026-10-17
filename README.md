# MoE-Health

Code for **MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction**
(Xiaoyang Wang and Christopher C. Yang, ACM BCB 2025,
[doi:10.1145/3765612.3767207](https://doi.org/10.1145/3765612.3767207)).

MoE-Health predicts in-hospital mortality, long length of stay (> 7 days) and 30-day readmission from the
first 48 hours of an admission. A patient may have any subset of three modalities: EHR time series,
radiology reports and chest X-rays (CXR). Each modality has its own encoder, and an absent modality is
replaced by a learned missing embedding. Multi-head attention fuses the modality tokens. A gating network
then routes each sample to the top-2 of 7 experts, one per modality combination, and a
coefficient-of-variation loss keeps the expert load balanced.

## Data

All datasets are on PhysioNet and require credentialed access.

| Dataset | Version | Used for |
|---|---|---|
| MIMIC-IV | 1.0 | admissions, patients, ICU stays, chart / lab / procedure / input events |
| MIMIC-CXR-JPG | 2.0.0 | chest X-ray images and metadata |
| MIMIC-IV-Note | 2.2 | radiology reports |
| HAIM-MIMIC-MM (`haim-multimodal`) | 1.0.1 | `haim_mimiciv_key_ids.csv`, the cohort key ids |

Set the four source paths in `run/conf/paths/default.yaml`. Alternatively, copy that file to
`run/conf/paths/local.yaml`, which git ignores. With a local file, add `paths=local` to every script
call, or `--paths local` for `launch.py` and `collect_results.py`.

The scripts read uncompressed tables, so gunzip the `.csv.gz` files first. Everything derived from the
data is written under `outputs/`, which is also git-ignored. The Bio_ClinicalBERT and torchxrayvision
weights download on first use.

## Setup

```bash
bash run/pipeline/setup_env.sh        # uv venv (Python 3.11), torch 2.7.1 + CUDA 12.6, this package (editable)
.venv/bin/python -m pytest -q         # unit and smoke tests on synthetic data, no MIMIC access needed
```

`requirements.lock.txt` lists the package versions used for the experiments.

## Pipeline

```bash
# 1. cohort: 31,088 admissions, 5 patient-level 80/10/10 splits (checked against the counts in paper Fig. 3)
.venv/bin/python run/pipeline/build_cohort.py

# 2. features
.venv/bin/python run/pipeline/extract_ehr.py                         # 48 h hourly EHR, paper variable list
.venv/bin/python run/pipeline/extract_ehr.py ehr.version=v2          # extended ICU variable set (tuned recipe)
.venv/bin/python run/pipeline/extract_notes.py notes.device=cuda:0   # Bio_ClinicalBERT report embeddings
.venv/bin/python run/pipeline/extract_cxr.py cxr.device=cuda:0       # DenseNet-121 CXR features
.venv/bin/python run/pipeline/extract_history.py                     # prior-utilisation codes (tuned recipe)
.venv/bin/python run/pipeline/extract_notes_tfidf.py                 # TF-IDF report features (tuned recipe)

# 3. one run
.venv/bin/python run/train.py model=moe_health task=mortality seed=0                     # paper settings
.venv/bin/python run/train.py model=moe_health +recipe=moe_final task=mortality seed=0   # tuned recipe

# 4. all experiments, then the summary tables
.venv/bin/python run/pipeline/launch.py main table2 ablation --gpus 0 1 2 3 --jobs-per-gpu 3
.venv/bin/python run/pipeline/collect_results.py     # -> outputs/results/{runs.csv,summary.md}
```

`launch.py` skips runs that already have a `metrics.json`. Before starting a job it waits until the GPU
has at least 8 GB of free memory, and it limits each job to 4 CPU threads.

| Grid | Runs | Content |
|---|---|---|
| `main` | 105 | MoE-Health with the tuned recipe and with the paper settings: both cohorts, 3 tasks, 5 seeds. Also unimodal EHR / notes / CXR models on the corrected cohort |
| `table2` | 30 | LOS with the modality subsets E, T, I, E+T, E+I, T+I (E+T+I is in `main`) |
| `ablation` | 45 | mortality: the paper's Table 3 ablations and further variants |

## Cohorts and labels

Select the cohort with `cohort=corrected` (the default) or `cohort=faithful`.

- **`faithful`** uses the paper's cohort of 31,088 admissions and its label definitions:
  - death is `discharge_location == DIED`;
  - long LOS is LOS > 7 days;
  - readmission is searched among the cohort's own admissions.
- **`corrected`** (28,403 admissions) changes three things:
  - It drops the 2,685 admissions that end with discharge or death within the 48 h input window, so
    every outcome occurs after prediction time.
  - Mortality comes from `hospital_expire_flag`.
  - Readmission is any admission within 30 days of discharge in the full MIMIC-IV admissions table,
    with in-hospital deaths excluded.

## Configuration

| Group | Options |
|---|---|
| `model.*` (`run/conf/model/moe_health.yaml`) | `experts` (all7, observed4), `pretrain` (superset, exact, generic, none), `top_k`, `gating` (learned, uniform), `missing` (learned, zeros), `expert_output` (repr, logit), `balance` (cv, switch, none), `alpha` |
| `data.*` (`run/conf/train.yaml`) | `modalities` (any subset of ETI), `ehr_version` (v1, v2), `ehr_stats`, `ehr_delta`, `note_variant` (cls, tok, tfidf, tok_tfidf), `note_logit`, `cxr_variant` (first, mean), `history` |
| `train.*` (`run/conf/train.yaml`) | optimiser, early stopping, modality dropout, per-group learning-rate multipliers |

The tuned recipe in `run/conf/recipe/moe_final.yaml` changes only the training settings and the EHR
and note inputs. The architecture and the loss are the paper's.

## Outputs

Each run writes to `outputs/runs/{exp}/{model}{tag}/{cohort}/{task}/seed{k}/`:
- `metrics.json`: AUROC, AUPRC, and F1 at the threshold that maximises F1 on validation. It also has
  per-combination metrics and routing statistics.
- `predictions.npz`: per-sample predictions. These are credentialed data.
- `config.yaml`: the resolved config.
- `train.log`.

## Layout

```
src/
  data_module/     cohort rules, EHR variable spec + DuckDB extraction, note sections, TF-IDF note features,
                   history codes, dataset
  model_module/    encoders, MoE layer (gate, top-k, balance loss), MoE-Health, unimodal model
  trainer_module/  training loop, MoE-Health stages 0 / A / B / C, unimodal training
  metrics_module/  AUROC, AUPRC, F1 at the validation-optimal threshold
run/
  conf/            Hydra configs: train.yaml, pipeline.yaml, model/, recipe/, paths/
  pipeline/        build_cohort, extract_{ehr,notes,notes_tfidf,cxr,history}, launch, collect_results, setup_env
  train.py         one (method, cohort, task, seed) run
tests/             unit and smoke tests on synthetic data
```

## Citation

```bibtex
@inproceedings{wang2025moehealth,
  author    = {Wang, Xiaoyang and Yang, Christopher C.},
  title     = {MoE-Health: A Mixture of Experts Framework for Robust Multimodal Healthcare Prediction},
  booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology,
               and Health Informatics},
  series    = {BCB '25},
  year      = {2025},
  location  = {Philadelphia, PA, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3765612.3767207}
}
```
