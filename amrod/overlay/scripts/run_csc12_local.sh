#!/usr/bin/env bash
set -euo pipefail

# Cityscapes-C short-term protocol (AMROD Table 2): 12 corruptions, each seen
# once, continual, no reset. Mounts all 12 corruption directories.
#
#   bash scripts/run_csc12_local.sh GPU EXP_NAME CONFIG [SEED]

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
# Mount whole: annotations/*.json are relative symlinks into this parent.
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

# AMROD Table 2 ordering.
CORRUPTIONS=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
             brightness contrast elastic_transform pixelate jpeg_compression)

OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${HOST_OUT}/logs/${EXP}.log"
mkdir -p "$(dirname "${LOG}")"

STREAM="("
MOUNTS=()
for c in "${CORRUPTIONS[@]}"; do
  [[ -d "${CSC_ROOT}/${c}/leftImg8bit/val" ]] || { echo "ERROR: missing ${CSC_ROOT}/${c}/leftImg8bit/val" >&2; exit 2; }
  MOUNTS+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
  STREAM+="\"${c}_mtl\","
done
STREAM="${STREAM%,})"
[[ -f "${CITYSCAPES_ROOT}/annotations/instancesonly_filtered_gtFine_val.json" ]] || { echo "ERROR: missing detection GT" >&2; exit 2; }

echo "======================================================================"
echo "CITYSCAPES-C SHORT-TERM (12 corruptions once) — AMROD Table 2 protocol"
echo "EXP    : ${EXP}   GPU: ${GPU}   SEED: ${SEED}"
echo "CONFIG : ${CFG}"
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
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  "${MOUNTS[@]}" \
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
