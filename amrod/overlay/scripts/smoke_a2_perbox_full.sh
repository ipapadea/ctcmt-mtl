#!/usr/bin/env bash
set -euo pipefail
HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
cd "${HOST_REPO}"

python3 -m py_compile detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py scripts/test_a2_perbox_full.py

docker run --rm \
  --gpus '"device=0"' \
  --shm-size=4g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  -v "${HOST_REPO}:/workspace/amrod:ro" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" \
  python scripts/test_a2_perbox_full.py
