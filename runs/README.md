# Baseline runs (QD-DETR, MS-CLAP, this repository at rev 481d60b)

Both runs use the unmodified `src/` of this repository. Only text outputs are kept here: configs, scripts, logs, metrics, and predictions. Checkpoints (`*.pth`, `resume_state.pt`), console logs, and the Clotho-Moment validation predictions are left on the server.

| Run | Folder | Seeds | Test R1@0.7 |
|---|---|---|---|
| CASTELLA only | `castella_10seeds/` | 2023–2032 | 9.70 ± 1.45 (`castella_10seeds_summary.tsv`) |
| Clotho-Moment pre-training → CASTELLA fine-tuning | `qd-clotho-pretrain-20261007/` | 2023–2027 | in progress (see below) |

Official README values (5 runs, CASTELLA test): CASTELLA only 10.17 ± 0.86, with Clotho-Moment pre-training 13.85 ± 1.47.

## castella_10seeds/ (2026-08-03 to 08-04)

- `configs/castella_seed<seed>.yml`: `config.yml` with seed and results_dir changed.
- `results/seed<seed>/`: train/val logs, best validation predictions and metrics, test `submission.jsonl` and `submission_metrics.json`, and eval logs.

## qd-clotho-pretrain-20261007/ (started 2026-10-07 11:43Z)

Its `README.md` covers the setup, the resume wrapper (`qd_resumable.py`), and the GPU reservation script (`reserve.py`).

- `results/P_s<seed>/`: Clotho-Moment pre-training (200 epochs), with train/val logs and val metrics.
- `results/F_s<seed>/`: CASTELLA fine-tuning from the best P checkpoint, with `eval_val/` and `eval_test/` (each scored once).

Status at push (2026-10-08 02Z): seeds 2023–2026 are done. Seed 2027 is still in pre-training, at epoch 96 of 200.

| Seed | Val R1@0.7 | Test R1@0.5 | Test R1@0.7 | Test mAP |
|---|---|---|---|---|
| 2023 | 18.18 | 24.87 | 12.69 | 10.51 |
| 2024 | 23.01 | 27.17 | 14.63 | 12.97 |
| 2025 | 21.88 | 27.39 | 16.18 | 12.93 |
| 2026 | 23.86 | 28.73 | 16.85 | 13.60 |
