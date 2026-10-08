# Baseline runs (QD-DETR)

These are historical run artifacts committed in `010dff2` with the fork source at `481d60b`, before the official upstream fixes in `bf0e88f` (T2V attention mask) and `35fcbaf` (two CASTELLA timestamps). The source and annotations in the current checkout now include those fixes. Existing predictions, metrics, and checkpoints were **not** regenerated with the corrected source and annotations; the training-annotation fix in particular requires a new training run to assess its effect. Use `481d60b` to inspect the source and data associated with these artifacts.

This repository keeps configs, scripts, metrics, and predictions. Checkpoints (`*.pth`, `resume_state.pt`), console logs, and the Clotho-Moment validation predictions were not included in this upload. The fork retains its `val_path` training fix and added R@5 evaluator alongside the official upstream fixes.

| Run | Folder | Seeds | Test R1@0.7 |
|---|---|---|---|
| CASTELLA only | `castella_10seeds/` | 2023–2032 | 9.70 ± 1.45 (`castella_10seeds_summary.tsv`) |
| Clotho-Moment pre-training → CASTELLA fine-tuning | `qd-clotho-pretrain-20261007/` | 2023–2027 | in progress (see below) |
| M2D-CLAP features: Clotho-Moment pre-training → CASTELLA fine-tuning (481d60b) | `qd-m2d-20261008/` | 2023 | 25.58 (1,333-query subset; see its `README.md`) |

Official README values (5 runs, CASTELLA test): CASTELLA only 10.17 ± 0.86, with Clotho-Moment pre-training 13.85 ± 1.47.

## castella_10seeds/ (2026-08-03 to 08-04)

- `configs/castella_seed<seed>.yml`: `config.yml` with seed and results_dir changed.
- `results/seed<seed>/`: best validation predictions and metrics, test `submission.jsonl` and `submission_metrics.json`.

## qd-clotho-pretrain-20261007/ (started 2026-10-07 11:43Z)

Its `README.md` covers the setup, the resume wrapper (`qd_resumable.py`), and the GPU reservation script (`reserve.py`).

- `results/P_s<seed>/`: Clotho-Moment pre-training (200 epochs), with val metrics and completion markers where available.
- `results/F_s<seed>/`: CASTELLA fine-tuning from the best P checkpoint, with `eval_val/` and `eval_test/` (each scored once).

Status at push (2026-10-08 02Z): seeds 2023–2026 are done. Seed 2027 is still in pre-training, at epoch 96 of 200.

| Seed | Val R1@0.7 | Test R1@0.5 | Test R1@0.7 | Test mAP |
|---|---|---|---|---|
| 2023 | 18.18 | 24.87 | 12.69 | 10.51 |
| 2024 | 23.01 | 27.17 | 14.63 | 12.97 |
| 2025 | 21.88 | 27.39 | 16.18 | 12.93 |
| 2026 | 23.86 | 28.73 | 16.85 | 13.60 |

## qd-m2d-20261008/ (2026-10-08)

QD-DETR with M2D-CLAP audio and text features instead of MS-CLAP. Its `README.md` (Korean) covers the setup.

- `clotho_pretrain/`: Clotho-Moment pre-training, run with source `cbbb5be`. Clotho-Moment audio has no padding (all 60 frames), so the attention-mask fix does not change this computation.
- `castella_finetune_481d60b/`: CASTELLA fine-tuning from that checkpoint, run with source and annotations at `481d60b` (the same code as the MS-CLAP runs). Only queries with M2D features are used (train 2,158 / val 343 / test 1,333), so its test numbers are not directly comparable with the 1,347-query MS-CLAP table above.
