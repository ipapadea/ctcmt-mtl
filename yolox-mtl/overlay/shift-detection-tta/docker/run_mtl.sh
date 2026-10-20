#!/usr/bin/env bash
# Run one tools/test.py evaluation of a YOLOX-MTL config inside docker.
#
#   bash docker/run_mtl.sh GPU CONFIG CKPT OUT_DIR [extra tools/test.py args...]
#
# Repo, datasets and checkpoint are mounted read-only; only OUT_DIR is writable. Log: OUT_DIR/eval.log
# Env: IMAGE (default shift-detection-tta:latest), DATA_HOST (datasets root mounted at /workspace/data),
#      EXTRA_MOUNTS (docker -v args, e.g. targets of absolute symlinks inside DATA_HOST).
set -euo pipefail
GPU="${1:?GPU}"; CONFIG="${2:?config}"; CKPT="${3:?checkpoint}"; OUT="${4:?out dir}"; shift 4 || true
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${IMAGE:-shift-detection-tta:latest}"
DATA_HOST="${DATA_HOST:?datasets root, mounted read-only at /workspace/data}"
[[ -f "$REPO/$CONFIG" ]] || { echo "ERROR: missing $REPO/$CONFIG" >&2; exit 2; }
[[ -f "$CKPT" ]] || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 2; }
[[ -e "$OUT/eval.log" ]] && { echo "ERROR: $OUT/eval.log exists (not overwriting)" >&2; exit 3; }
mkdir -p "$OUT"
{ echo "config=$CONFIG"; echo "checkpoint=$CKPT"; echo "checkpoint_md5=$(md5sum "$CKPT" | cut -d' ' -f1)";
  echo "repo_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"; echo "start=$(date -Is)"; } > "$OUT/meta.txt"
# shellcheck disable=SC2086
docker run --rm --gpus "\"device=$GPU\"" --shm-size=16g --ipc=host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -v "$REPO":/workspace/shift-detection-tta:ro \
  -v "$DATA_HOST":/workspace/data:ro ${EXTRA_MOUNTS:-} \
  -v "$(dirname "$CKPT")":/workspace/ckpt:ro \
  -v "$OUT":/workspace/out \
  -w /workspace/shift-detection-tta \
  "$IMAGE" \
  python tools/test.py "$CONFIG" --checkpoint "/workspace/ckpt/$(basename "$CKPT")" \
    --work-dir /workspace/out "$@" > "$OUT/eval.log" 2>&1
echo "end=$(date -Is)" >> "$OUT/meta.txt"
