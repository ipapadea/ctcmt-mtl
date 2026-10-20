#!/usr/bin/env bash
# Run the shift-detection-tta container with GPU access.
#
# Bind mounts (all overrideable via env):
#   $REPO_ROOT   -> /workspace/shift-detection-tta     (this repo)
#   $DATA_HOST   -> /workspace/data                    (datasets: cityscapes, acdc, ...)
#   $WORK_DIRS   -> /workspace/shift-detection-tta/work_dirs  (checkpoints/logs)
#
# Usage:
#   docker/run.sh                                       # interactive shell
#   docker/run.sh python tools/train.py <config>        # one-shot command
#   docker/run.sh bash -c 'python -c "import mmdet, mmtrack; print(mmdet.__version__)"'
#
# Env vars:
#   IMAGE       image tag                (default: shift-detection-tta:latest)
#   DATA_HOST   host datasets directory  (default: <repo>/data — create it or set explicitly)
#   WORK_DIRS   host work_dirs directory (default: <repo>/work_dirs)
#   GPUS        --gpus argument          (default: all; e.g. '"device=0"' to pick GPU 0)
#   NAME        container name           (default: shift-detection-tta)
#   DETACH      set to 1 for -d          (default: interactive)
#   EXTRA_ARGS  extra docker run flags
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

IMAGE="${IMAGE:-shift-detection-tta:latest}"
DATA_HOST="${DATA_HOST:-$REPO_ROOT/data}"
WORK_DIRS="${WORK_DIRS:-$REPO_ROOT/work_dirs}"
GPUS="${GPUS:-all}"
NAME="${NAME:-shift-detection-tta}"

mkdir -p "$DATA_HOST" "$WORK_DIRS"

# Interactive by default; detach if requested.
if [[ "${DETACH:-0}" == "1" ]]; then
    RUN_FLAGS=(-d)
else
    RUN_FLAGS=(-it --rm)
fi

# Reuse the host user so files written into work_dirs/ aren't root-owned.
USER_FLAGS=(-u "$(id -u):$(id -g)"
            -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib
            -e XDG_CACHE_HOME=/tmp/.cache)

echo "[run] image=$IMAGE  repo=$REPO_ROOT  data=$DATA_HOST  work_dirs=$WORK_DIRS"

exec docker run \
    "${RUN_FLAGS[@]}" \
    --name "$NAME" \
    --gpus "$GPUS" \
    --shm-size=16g \
    --ipc=host \
    "${USER_FLAGS[@]}" \
    -v "$REPO_ROOT":/workspace/shift-detection-tta \
    -v "$DATA_HOST":/workspace/data \
    -v "$WORK_DIRS":/workspace/shift-detection-tta/work_dirs \
    -w /workspace/shift-detection-tta \
    ${EXTRA_ARGS:-} \
    "$IMAGE" \
    "$@"
