#!/usr/bin/env bash
# QD-DETR (original code rev 481d60b, the code of the MS-CLAP runs) CASTELLA fine-tuning with M2D-CLAP features, initialized from the M2D
# Clotho-Moment pretraining of the same seed (results/qd-m2d-clotho-pretrain-20261008), then val and
# test inference (test scored once per seed). One container per seed on GPU0.
# Start and resume with the same command: finished stages are skipped (DONE.json), an interrupted
# training continues from resume_state.pt, a running seed's container is left alone. A seed whose
# pretraining has no DONE.json is not started.
#   ./start_finetune.sh 2023 2024 2025
# Stop one seed: docker stop qd-m2d-castella-finetune-481d60b-s<seed>
set -euo pipefail
R="$(cd "$(dirname "$0")" && pwd)"
L=/home/soyeong/dcase2026/legacy/task6-qd-detr
P=/home/soyeong/dcase2026/results/qd-m2d-clotho-pretrain-20261008
FILL=/home/soyeong/dcase2026/results/qd-m2d-castella-fill-20261008
OG_AUDIO="/home/soyeong/dcase2026/A_Ogawa_v1(0923)/runs/20260924_full_extract/features/audio"
OG_TEXT="/home/soyeong/dcase2026/A_Ogawa_v1(0923)/runs/gapcfg_20261007/text_m2d/castella"
GPU0=GPU-a05bb29b-3af6-f296-441d-fff3dffa21bc
IMAGE=dcase26-task6:py271-cu128
[ $# -ge 1 ] || { echo "usage: $0 <seed> [<seed> ...]" >&2; exit 2; }
[ "$(git -C "$L" rev-parse --short HEAD)" = 481d60b ] || { echo "code checkout is not 481d60b" >&2; exit 1; }
[ -z "$(git -C "$L" status --porcelain -- src config.yml)" ] || { echo "src/ or config.yml of $L has local changes" >&2; exit 1; }
mkdir -p "$R/logs"
for s in "$@"; do
  NAME=qd-m2d-castella-finetune-481d60b-s$s
  if [ -f "$R/results/castella_finetune_m2d_s$s/eval_test/DONE.json" ]; then echo "seed $s: complete, skip"; continue; fi
  if [ ! -f "$P/results/clotho_pretrain_m2d_s$s/DONE.json" ]; then echo "seed $s: pretraining not finished, not started"; continue; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; then echo "$NAME is already running"; continue; fi
  log="$R/logs/finetune_s$s.log"
  inner="python $R/qd_resumable.py finetune --run-dir $R --pretrain-dir $P --seeds $s >> $log 2>&1; rc=\$?; echo \"EXIT_STATUS=\$rc \$(date -u +%FT%TZ)\" >> $log; exit \$rc"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "device=$GPU0" --user 1002:1002 --cpus 4 --shm-size 8g --network none \
    -e HOME=/tmp -e TQDM_MININTERVAL=60 \
    -v "$L:$L:ro" -v "$P:$P:ro" -v "$FILL:$FILL:ro" -v "$OG_AUDIO:$OG_AUDIO:ro" -v "$OG_TEXT:$OG_TEXT:ro" -v "$R:$R:rw" \
    -w "$L" --entrypoint "" "$IMAGE" bash -c "$inner" >/dev/null
  echo "$(date -u +%FT%TZ) started $NAME on GPU0 (seed $s)" | tee -a "$R/logs/start.log"
done
