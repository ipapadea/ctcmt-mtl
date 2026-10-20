#!/usr/bin/env bash
set -euo pipefail

# Offline CTAOD source-feature statistics.
#
#   bash scripts/collect_ctaod_source_stats_local.sh [GPU] [NUM_IMAGES]
#
# Runs ONCE on the Cityscapes source training split with the canonical source
# checkpoint.  The statistics file is the only source-derived asset used by the
# CTAOD baseline; no source image is accessed during online adaptation.

GPU="${1:-0}"
NUM_IMAGES="${2:-2000}"
SEED="${3:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
# needs shapely + cityscapesscripts to parse gtFine polygons
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod-train:latest}"

SOURCE_WEIGHTS_HOST="${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
EXPECTED_MD5="5f63c7578cb925552fd1b9d69980937a"
ACTUAL_MD5="$(md5sum "${SOURCE_WEIGHTS_HOST}" | awk '{print $1}')"
[[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]] || {
  echo "ERROR: source checkpoint MD5 mismatch (${ACTUAL_MD5})" >&2; exit 2; }
echo "source checkpoint MD5 OK: ${ACTUAL_MD5}"

OUT_PT="/workspace/output/ctaod_stats/newsrc_ctaod_source_stats.pt"
LOG="${HOST_OUT}/logs/ctaod_source_stats.log"
mkdir -p "$(dirname "${LOG}")" "${HOST_OUT}/ctaod_stats"

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
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/collect_ctaod_source_stats.py \
      --config-file detectron2/configs/Cityscapes/newsrc_ctaod_acdc.yaml \
      --dataset cityscapes_fine_instance_seg_train \
      --num-images ${NUM_IMAGES} \
      --seed ${SEED} \
      --output ${OUT_PT} \
      OUTPUT_DIR /workspace/output/ctta_acdc/ctaod_source_stats
  " 2>&1 | tee "${LOG}"

echo "DONE -> ${HOST_OUT}/ctaod_stats/$(basename ${OUT_PT})"
