#!/usr/bin/env bash
set -euo pipefail

# Source-only (frozen model) evaluation on Cityscapes-C, on this machine.
#
#   bash scripts/run_source_only_cs_c_local.sh [GPU] [EXP] [CONFIG]

GPU="${1:-0}"
EXP="${2:-source_only_pfn_cs_c}"
CFG="${3:-detectron2/configs/Cityscapes/source_only_pfn_cs_c.yaml}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
# Mount this root whole: annotations/*.json are relative symlinks into its parent,
# so mounting the annotations subdir alone leaves them dangling.
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

CORRUPTIONS=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
             brightness contrast elastic_transform pixelate jpeg_compression)

OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${HOST_OUT}/logs/${EXP}.log"
mkdir -p "$(dirname "${LOG}")"

MOUNTS=()
for c in "${CORRUPTIONS[@]}"; do
  [[ -d "${CSC_ROOT}/${c}/leftImg8bit/val" ]] || { echo "ERROR: missing ${CSC_ROOT}/${c}/leftImg8bit/val" >&2; exit 2; }
  MOUNTS+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
done
[[ -d "${CITYSCAPES_ROOT}/gtFine/val" ]] || { echo "ERROR: missing ${CITYSCAPES_ROOT}/gtFine/val" >&2; exit 2; }
[[ -f "${CITYSCAPES_ROOT}/annotations/instancesonly_filtered_gtFine_val.json" ]] || { echo "ERROR: missing detection GT under ${CITYSCAPES_ROOT}/annotations" >&2; exit 2; }

echo "======================================================================"
echo "SOURCE-ONLY (no adaptation) — CITYSCAPES-C, 12 corruptions"
echo "EXP    : ${EXP}"
echo "GPU    : ${GPU}"
echo "CONFIG : ${CFG}"
echo "ORDER  : ${CORRUPTIONS[*]}"
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
      OUTPUT_DIR ${OUT}
  " 2>&1 | tee "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
