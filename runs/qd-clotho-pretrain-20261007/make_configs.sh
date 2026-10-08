#!/usr/bin/env bash
# Derive per-seed configs from the official baseline configs (legacy/task6-qd-detr, rev 481d60b).
# P_s<seed>: config_pretraining.yml (Clotho-Moment); F_s<seed>: config.yml (CASTELLA, same file the
# 2026-08-03 10-seed CASTELLA-only run used).  Changed keys only: seed, results_dir, and for P the
# Clotho feature dirs (official Zenodo 17129257 MS-CLAP features, mounted read-only at /feat/clotho).
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
L=/home/soyeong/dcase2026/legacy/task6-qd-detr
mkdir -p "$R/configs"
for s in "$@"; do
  sed -e "s|^seed: .*|seed: $s|" -e "s|^results_dir: .*|results_dir: $R/results/P_s$s|" \
      -e "s|^a_feat_dir: .*|a_feat_dir: /feat/clotho/clap|" -e "s|^t_feat_dir: .*|t_feat_dir: /feat/clotho/clap_text|" \
      "$L/config_pretraining.yml" > "$R/configs/P_s$s.yml"
  sed -e "s|^seed: .*|seed: $s|" -e "s|^results_dir: .*|results_dir: $R/results/F_s$s|" \
      "$L/config.yml" > "$R/configs/F_s$s.yml"
done
