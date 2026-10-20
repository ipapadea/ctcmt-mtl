#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU}"
SEED="${2:-0}"
HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/data/ilias/panoptic_fpn/output}"
CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/ilias/cityscapes_pfn}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-/data/vgcmt/datasets/cityscapes/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
LOG_ROOT="${LOG_ROOT:-/home/ilias}"

CFG="detectron2/configs/Cityscapes/ctcmt_e8a2_v2_perbox_full_ctcr_no_ctpv.yaml"
EXP="ctcr_A2_v2_perbox_full_no_ctpv_fogx10_seed${SEED}"
OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${LOG_ROOT}/${EXP}.log"
STREAM='("fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl")'

cd "${HOST_REPO}"
rm -rf "${HOST_OUT}/ctta_acdc/${EXP}"

docker run --rm \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NVIDIA_TF32_OVERRIDE=0 \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  -v "${HOST_REPO}:/workspace/amrod:ro" \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  -v "${CITYSCAPES_ANN_ROOT}:/datasets/cityscapes/annotations:ro" \
  -v "${CSC_ROOT}/fog:/datasets/fog:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" \
  python detectron2/tools/train_net.py \
    --config-file "${CFG}" \
    --eval-only \
    --num-gpus 1 \
    OUTPUT_DIR "${OUT}" \
    SEED "${SEED}" \
    DATASETS.TEST "${STREAM}" \
  2>&1 | tee "${LOG}"
