#!/usr/bin/env bash
set -euo pipefail

# Run ONE CT-CR variant on the ACDC continual stream (fog -> night -> rain -> snow)
# on THIS machine (Cronus, workspace under /media/ilias/DATA/ilias).
#
# Usage:
#   bash scripts/run_ctcr_acdc_local.sh GPU EXP_NAME CONFIG [SEED]

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
ACDC_ROOT="${ACDC_ROOT:-/media/ilias/DATA/ilias/acdc}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes_pfn}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-/media/ilias/DATA/ilias/cityscapes/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${LOG_ROOT:-${HOST_OUT}/logs}/${EXP}.log"
mkdir -p "$(dirname "${LOG}")"

for d in "${HOST_REPO}" "${HOST_OUT}" "${ACDC_ROOT}/rgb_anon" "${ACDC_ROOT}/gt" \
         "${ACDC_ROOT}/gt_detection" "${CITYSCAPES_ROOT}/gtFine" "${CITYSCAPES_ANN_ROOT}"; do
  [[ -e "${d}" ]] || { echo "ERROR: required path not found: ${d}" >&2; exit 2; }
done

EXTRA_OPTS="${EXTRA_OPTS:-}"
# det-only baselines use acdc_*, seg-only use acdc_*_semseg, MTL uses acdc_*_mtl
STREAM="${STREAM:-(\"acdc_fog_mtl\",\"acdc_night_mtl\",\"acdc_rain_mtl\",\"acdc_snow_mtl\")}"

echo "======================================================================"
echo "CT-CR — ACDC CONTINUAL (fog -> night -> rain -> snow), no reset"
echo "EXP    : ${EXP}"
echo "GPU    : ${GPU}"
echo "SEED   : ${SEED}"
echo "CONFIG : ${CFG}"
echo "TF32   : OFF"
echo "======================================================================"

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
  -v "${ACDC_ROOT}:/datasets/ACDC:ro" \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  -v "${CITYSCAPES_ANN_ROOT}:/datasets/annotations:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only \
      --num-gpus 1 \
      OUTPUT_DIR ${OUT} \
      SEED ${SEED} \
      DATASETS.TEST '${STREAM}' ${EXTRA_OPTS}
  " 2>&1 | tee "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
echo "OUT : ${HOST_OUT}/ctta_acdc/${EXP}"
