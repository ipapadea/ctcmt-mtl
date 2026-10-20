#!/usr/bin/env bash
# Build the shift-detection-tta docker image.
#
# Usage:
#   docker/build.sh              # tag = shift-detection-tta:latest
#   IMAGE=my/tag docker/build.sh # custom tag
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-shift-detection-tta:latest}"

echo "[build] image=$IMAGE  context=$REPO_ROOT"
cd "$REPO_ROOT"
docker build \
    --file docker/Dockerfile \
    --tag "$IMAGE" \
    "$@" \
    .
