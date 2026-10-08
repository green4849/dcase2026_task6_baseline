#!/usr/bin/env bash
# QD-DETR Clotho-Moment pretraining configs with M2D-CLAP features, from the official
# config_pretraining.yml at fork rev cbbb5be (legacy/task6-qd-detr-cbbb5be).
# Changed keys only: seed, results_dir, a_feat_dir, t_feat_dir. Dimensions stay 768 (both feature
# sets are 768-d), clip_length stays 1 (features are 1 s mean-pooled, 60 frames per 60 s clip).
#   ./make_configs.sh 2023 2024 2025
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
L=/home/soyeong/dcase2026/legacy/task6-qd-detr-cbbb5be
mkdir -p "$R/configs"
for s in "$@"; do
  sed -e "s|^seed: .*|seed: $s|" -e "s|^results_dir: .*|results_dir: $R/results/clotho_pretrain_m2d_s$s|" \
      -e "s|^a_feat_dir: .*|a_feat_dir: /feat/m2d/clotho_audio|" -e "s|^t_feat_dir: .*|t_feat_dir: /feat/m2d/clotho_text|" \
      "$L/config_pretraining.yml" > "$R/configs/clotho_pretrain_m2d_s$s.yml"
done
