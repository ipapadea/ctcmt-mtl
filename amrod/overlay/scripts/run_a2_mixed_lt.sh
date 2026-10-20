#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?GPU}"
SEED="${2:?SEED}"

HOST_REPO="${HOST_REPO:-/home/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/data/ilias/panoptic_fpn/output}"
CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
LOG_ROOT="${LOG_ROOT:-/home/ilias}"

if [[ "$(hostname)" == "Cronus" ]]; then
  CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"
else
  CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/ilias/cityscapes_pfn}"
fi
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-${CITYSCAPES_ROOT}/annotations}"

CFG="detectron2/configs/Cityscapes/ctcmt_e8a2_v2_perbox_full_ctcr_no_ctpv.yaml"
EXP="ctcr_A2_v2_perbox_full_no_ctpv_csc_mixed_lt_x10_seed${SEED}"
OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${LOG_ROOT}/${EXP}.log"

STREAM='("fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl","fog_mtl","motion_blur_mtl","snow_mtl","brightness_mtl","defocus_blur_mtl")'

cd "${HOST_REPO}"
test -f "${CFG}"
test -d "${CITYSCAPES_ROOT}/gtFine/val"
test -f "${CITYSCAPES_ANN_ROOT}/instancesonly_filtered_gtFine_val.json"
for c in fog motion_blur snow brightness defocus_blur; do test -d "${CSC_ROOT}/${c}"; done

rm -rf "${HOST_OUT}/ctta_acdc/${EXP}"

MOUNTS=(
  -v "${HOST_REPO}:/workspace/amrod:ro"
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro"
  -v "${CSC_ROOT}/fog:/datasets/fog:ro"
  -v "${CSC_ROOT}/motion_blur:/datasets/motion_blur:ro"
  -v "${CSC_ROOT}/snow:/datasets/snow:ro"
  -v "${CSC_ROOT}/brightness:/datasets/brightness:ro"
  -v "${CSC_ROOT}/defocus_blur:/datasets/defocus_blur:ro"
  -v "${HOST_OUT}:/workspace/output"
)
if [[ "${CITYSCAPES_ANN_ROOT}" != "${CITYSCAPES_ROOT}/annotations" ]]; then
  MOUNTS+=( -v "${CITYSCAPES_ANN_ROOT}:/datasets/cityscapes/annotations:ro" )
fi

echo "A2 mixed LT seed=${SEED} GPU=${GPU} host=$(hostname)"
docker run --rm \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NVIDIA_TF32_OVERRIDE=0 \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  "${MOUNTS[@]}" \
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
