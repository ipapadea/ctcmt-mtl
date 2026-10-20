#!/usr/bin/env bash
set -euo pipefail

# Evaluate a SOURCE checkpoint (no adaptation) on this machine.
#
#   bash scripts/eval_source_ckpt_local.sh GPU NAME CKPT clean|acdc
#
# CKPT is a path inside the container, i.e. under /workspace/output.
# Example:
#   bash scripts/eval_source_ckpt_local.sh 0 src_new_48k_w10 \
#     /workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_0047999.pth clean
#
# Parse the result with:  python3 scripts/gather_eval.py LOG

GPU="${1:?GPU id required}"
NAME="${2:?run name required}"
CKPT="${3:?checkpoint path (container-side) required}"
SUITE="${4:-clean}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
ACDC_ROOT="${ACDC_ROOT:-/media/ilias/DATA/ilias/acdc}"
# Not cityscapes_pfn: its contents are symlinks to another host and dangle here.
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-/media/ilias/DATA/ilias/coco_annotations}"
CFG="${CFG:-detectron2/configs/Cityscapes/panoptic_fpn_R_50.yaml}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

case "${SUITE}" in
  clean) STREAM='("cityscapes_val_mtl",)' ;;
  acdc)  STREAM='("acdc_fog_mtl","acdc_night_mtl","acdc_rain_mtl","acdc_snow_mtl",)' ;;
  *) echo "ERROR: suite must be clean|acdc, got ${SUITE}" >&2; exit 2 ;;
esac

LOG="${HOST_OUT}/logs/eval_${NAME}.log"
mkdir -p "$(dirname "${LOG}")"

echo "======================================================================"
echo "SOURCE EVAL  ${NAME}   suite=${SUITE}  gpu=${GPU}"
echo "CKPT : ${CKPT}"
echo "LOG  : ${LOG}"
echo "======================================================================"

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
  -v "${ACDC_ROOT}:/datasets/ACDC:ro" \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  -v "${CITYSCAPES_ANN_ROOT}:/datasets/annotations:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only --num-gpus 1 \
      MODEL.WEIGHTS ${CKPT} \
      OUTPUT_DIR /workspace/output/eval/${NAME} \
      DATASETS.TEST '${STREAM}'
  " > "${LOG}" 2>&1

echo "DONE  evals=$(grep -c 'in csv format' "${LOG}")  tracebacks=$(grep -c Traceback "${LOG}")"
