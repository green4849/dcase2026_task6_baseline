#!/usr/bin/env bash
# QD-DETR (fork rev cbbb5be) Clotho-Moment pretraining from random initialization with M2D-CLAP
# features on GPU0, one container per seed so seeds can run side by side
# (2026-10-08 08:1xZ, user request: run 2024 and 2025 in parallel; one job uses ~38% SM).
# Start and resume with the same command: a finished seed is skipped (DONE.json), an interrupted
# seed continues from resume_state.pt, a running seed's container is left alone.
#   ./start_pretrain.sh 2024 2025
# Stop one seed: docker stop qd-m2d-clotho-pretrain-s<seed>   (rerun this script to continue)
# Seed 2023 ran in the earlier sequential container qd-m2d-clotho-pretrain (logs/pretrain.log,
# script kept as logs/start_pretrain.sequential.sh.bak).
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
L=/home/soyeong/dcase2026/legacy/task6-qd-detr-cbbb5be
OG="/home/soyeong/dcase2026/A_Ogawa_v1(0923)/runs"
AUDIO="$OG/clotho_extract_20260925/features/audio"     # m2d2025-p80x2-timeframes-meanpool1s-v1
TEXT="$OG/gapcfg_20261007/text_m2d/clotho"            # m2dclap2025-p80x2-bert-last-hidden-state-v1
GPU0=GPU-a05bb29b-3af6-f296-441d-fff3dffa21bc
IMAGE=dcase26-task6:py271-cu128
[ $# -ge 1 ] || { echo "usage: $0 <seed> [<seed> ...]" >&2; exit 2; }
if [ "$(docker inspect -f '{{.State.Running}}' qd-m2d-clotho-pretrain 2>/dev/null)" = "true" ]; then
  echo "the sequential container qd-m2d-clotho-pretrain is still running; stop it first" >&2; exit 1
fi
[ "$(git -C "$L" rev-parse --short HEAD)" = cbbb5be ] || { echo "code checkout is not cbbb5be" >&2; exit 1; }
mkdir -p "$R/logs"
for s in "$@"; do
  NAME=qd-m2d-clotho-pretrain-s$s
  if [ -f "$R/results/clotho_pretrain_m2d_s$s/DONE.json" ]; then echo "seed $s: DONE, skip"; continue; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; then echo "$NAME is already running"; continue; fi
  log="$R/logs/pretrain_s$s.log"
  inner="python $R/qd_resumable.py train --config $R/configs/clotho_pretrain_m2d_s$s.yml >> $log 2>&1; rc=\$?; echo \"EXIT_STATUS=\$rc \$(date -u +%FT%TZ)\" >> $log; exit \$rc"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "device=$GPU0" --user 1002:1002 --cpus 4 --shm-size 8g --network none \
    -e HOME=/tmp -e TQDM_MININTERVAL=60 \
    -v "$L:$L:ro" -v "$AUDIO:/feat/m2d/clotho_audio:ro" -v "$TEXT:/feat/m2d/clotho_text:ro" -v "$R:$R:rw" \
    -w "$L" --entrypoint "" "$IMAGE" bash -c "$inner" >/dev/null
  echo "$(date -u +%FT%TZ) started $NAME on GPU0 (seed $s)" | tee -a "$R/logs/start.log"
done
