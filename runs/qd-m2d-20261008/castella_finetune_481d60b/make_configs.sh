#!/usr/bin/env bash
# QD-DETR CASTELLA fine-tuning configs with M2D-CLAP features, from the official config.yml at the
# original code rev 481d60b (legacy/task6-qd-detr, the code of the MS-CLAP runs). Changed keys only: seed, results_dir, a_feat_dir,
# t_feat_dir, train_path, val_path, test_path. Dimensions stay 768 and clip_length stays 1
# (1 s mean-pooled features, frames == annotation duration).
#   ./make_configs.sh 2023 2024 2025
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
L=/home/soyeong/dcase2026/legacy/task6-qd-detr
mkdir -p "$R/configs"
for s in "$@"; do
  sed -e "s|^seed: .*|seed: $s|" -e "s|^results_dir: .*|results_dir: $R/results/castella_finetune_m2d_s$s|" \
      -e "s|^a_feat_dir: .*|a_feat_dir: $R/features/audio|" -e "s|^t_feat_dir: .*|t_feat_dir: $R/features/text|" \
      -e "s|^train_path: .*|train_path: $R/data/castella_train_m2d.jsonl|" -e "s|^val_path: .*|val_path: $R/data/castella_val_m2d.jsonl|" \
      -e "s|^test_path: .*|test_path: $R/data/castella_test_m2d.jsonl|" \
      "$L/config.yml" > "$R/configs/castella_finetune_m2d_s$s.yml"
done
