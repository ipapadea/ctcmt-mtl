#!/usr/bin/env bash
set -euo pipefail

# Run ONE A/B/C CT-CR variant on the Cityscapes-C mixed long-term stream:
#
#   (fog -> motion_blur -> snow -> brightness -> defocus_blur) x 10
#
# Same CTCMT model instance is kept throughout DATASETS.TEST; no reset is
# requested between domains. Uses *_mtl registrations so both bbox and
# semantic-segmentation evaluators are available.
#
# Usage:
#   bash scripts/run_ctcr_mixed_lt_one.sh GPU EXP_NAME CONFIG [SEED]
#
# Paths can be overridden for Cronus without editing this file:
#   HOST_REPO=/path/to/AMROD \
#   HOST_OUT=/path/to/output \
#   CSC_ROOT=/path/to/cityscapes_c_amrod \
#   CITYSCAPES_ROOT=/path/to/cityscapes_pfn \
#   CITYSCAPES_ANN_ROOT=/path/to/cityscapes/annotations \
#   bash scripts/run_ctcr_mixed_lt_one.sh ...

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/data/ilias/panoptic_fpn/output}"
CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
#CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/ilias/cityscapes_pfn}"
#DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-${CITYSCAPES_ROOT}/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${LOG_ROOT:-/home/ilias}/${EXP}.log"

# 5-domain cycle repeated exactly 10 times = 50 continual evaluations.
STREAM='("fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl")'

for d in \
  "${HOST_REPO}" \
  "${HOST_OUT}" \
  "${CSC_ROOT}/fog" \
  "${CSC_ROOT}/motion_blur" \
  "${CSC_ROOT}/snow" \
  "${CSC_ROOT}/brightness" \
  "${CSC_ROOT}/defocus_blur" \
  "${CITYSCAPES_ROOT}/gtFine/val" \
  "${CITYSCAPES_ANN_ROOT}"
do
  if [[ ! -e "${d}" ]]; then
    echo "ERROR: required path not found: ${d}" >&2
    exit 2
  fi
done

if [[ ! -f "${CITYSCAPES_ANN_ROOT}/instancesonly_filtered_gtFine_val.json" ]]; then
  echo "ERROR: missing Cityscapes detection GT:" >&2
  echo "  ${CITYSCAPES_ANN_ROOT}/instancesonly_filtered_gtFine_val.json" >&2
  exit 2
fi

echo "======================================================================"
echo "CT-CR A/B/C — CITYSCAPES-C MIXED LONG-TERM"
echo "======================================================================"
echo "EXP        : ${EXP}"
echo "GPU        : ${GPU}"
echo "SEED       : ${SEED}"
echo "CONFIG     : ${CFG}"
echo "CYCLE      : fog -> motion_blur -> snow -> brightness -> defocus_blur"
echo "ROUNDS     : 10"
echo "EVALS      : 50"
echo "RESET      : none"
echo "TF32       : OFF"
echo "HOST_REPO  : ${HOST_REPO}"
echo "HOST_OUT   : ${HOST_OUT}"
echo "CSC_ROOT   : ${CSC_ROOT}"
echo "CS GT ROOT : ${CITYSCAPES_ROOT}"
echo "CS ANN     : ${CITYSCAPES_ANN_ROOT}"
echo "======================================================================"

# Avoid mixing partial results from a previous failed run.
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
  -v "${CSC_ROOT}/fog:/datasets/fog:ro" \
  -v "${CSC_ROOT}/motion_blur:/datasets/motion_blur:ro" \
  -v "${CSC_ROOT}/snow:/datasets/snow:ro" \
  -v "${CSC_ROOT}/brightness:/datasets/brightness:ro" \
  -v "${CSC_ROOT}/defocus_blur:/datasets/defocus_blur:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only \
      --num-gpus 1 \
      OUTPUT_DIR ${OUT} \
      SEED ${SEED} \
      DATASETS.TEST '${STREAM}'
  " 2>&1 | tee "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
echo "OUT : ${HOST_OUT}/ctta_acdc/${EXP}"
