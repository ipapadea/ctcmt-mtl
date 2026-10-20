#!/usr/bin/env bash
# Run one shift-detection-tta evaluation on the remote machine ("leviathan") inside docker.
# The repo, datasets and checkpoints are mounted READ-ONLY; only the work dir is writable.
#
#   bash docker/run_leviathan_eval.sh GPU CONFIG CKPT EXP [extra tools/test.py args...]
#
#   GPU     0 or 1
#   CONFIG  path relative to the repo, e.g. configs/continuous/no_adap_yolox/source_only_acdc4.py
#   CKPT    path of the checkpoint as seen from the host
#   EXP     name of the run; output goes to $OUT_ROOT/EXP (log: $OUT_ROOT/EXP/eval.log)
#
# Env: DATA_HOST (required), REPO, OUT_ROOT, IMAGE
set -euo pipefail

GPU="${1:?GPU id}"; CONFIG="${2:?config}"; CKPT="${3:?checkpoint}"; EXP="${4:?experiment name}"
shift 4 || true

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_HOST="${DATA_HOST:?datasets root}"
OUT_ROOT="${OUT_ROOT:-$HOME/work_dirs}"
IMAGE="${IMAGE:-shift-detection-tta:cu113}"

[[ -f "$REPO/$CONFIG" ]] || { echo "ERROR: missing $REPO/$CONFIG" >&2; exit 2; }
[[ -f "$CKPT" ]] || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 2; }
OUT="$OUT_ROOT/$EXP"
[[ -e "$OUT" ]] && { echo "ERROR: $OUT already exists (not overwriting)" >&2; exit 3; }
mkdir -p "$OUT"
{ echo "config=$CONFIG"; echo "checkpoint=$CKPT"; echo "checkpoint_md5=$(md5sum "$CKPT" | cut -d' ' -f1)";
  echo "repo_copy=$(cat "$REPO/PROVENANCE.txt" 2>/dev/null | head -1)"; echo "start=$(date -Is)"; } > "$OUT/meta.txt"

docker run --rm --gpus "\"device=$GPU\"" --shm-size=16g --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "$REPO":/workspace/shift-detection-tta:ro \
  -v "$DATA_HOST":/workspace/data:ro \
  -v "$(dirname "$CKPT")":/workspace/ckpt:ro \
  -v "$OUT":/workspace/out \
  -w /workspace/shift-detection-tta \
  "$IMAGE" \
  python tools/test.py "$CONFIG" --checkpoint "/workspace/ckpt/$(basename "$CKPT")" \
    --work-dir /workspace/out "$@" 2>&1 | tee "$OUT/eval.log"

echo "end=$(date -Is)" >> "$OUT/meta.txt"
