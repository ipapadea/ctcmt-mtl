#!/usr/bin/env bash
set -euo pipefail

# Train a Cityscapes SOURCE model on THIS machine (Cronus), in the amrod:latest
# container. Unlike the adaptation runners this one does NOT wipe the output
# directory, because a 60k-iteration run is worth resuming.
#
# Usage:
#   bash scripts/train_source_pfn_local.sh [GPU] [EXP] [CONFIG]
#   RESUME=1 bash scripts/train_source_pfn_local.sh 0        # continue a stopped run
#   EXTRA_OPTS="SOLVER.MAX_ITER 20" bash scripts/train_source_pfn_local.sh 0 smoke
#
# Defaults train panoptic_fpn_R_50_segw1.yaml on GPU 0.

GPU="${1:-0}"
EXP="${2:-panoptic_fpn_R50_cityscapes_segw1}"
CFG="${3:-detectron2/configs/Cityscapes/panoptic_fpn_R_50_segw1.yaml}"
SEED="${SEED:-0}"
RESUME="${RESUME:-0}"
TF32="${TF32:-0}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
# NOT cityscapes_pfn: its leftImg8bit/annotations are symlinks to another host
# and dangle inside the container. This root holds the real train split.
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
# amrod:latest plus shapely, which source training needs and inference does not.
# Build once: docker build -f Dockerfile.train -t amrod-train:latest .
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod-train:latest}"

OUT="/workspace/output/${EXP}"
HOST_OUT_DIR="${HOST_OUT}/${EXP}"
LOG="${LOG_ROOT:-${HOST_OUT}/logs}/${EXP}.log"
mkdir -p "$(dirname "${LOG}")" "${HOST_OUT_DIR}" "${HOST_OUT}/.cache"

for d in "${HOST_REPO}/${CFG}" "${HOST_OUT}" \
         "${CITYSCAPES_ROOT}/leftImg8bit/train" \
         "${CITYSCAPES_ROOT}/gtFine/train"; do
  [[ -e "${d}" ]] || { echo "ERROR: required path not found: ${d}" >&2; exit 2; }
done

# Never silently restart on top of an existing run's checkpoints.
if [[ "${RESUME}" != "1" ]] && compgen -G "${HOST_OUT_DIR}/model_*.pth" > /dev/null; then
  echo "ERROR: ${HOST_OUT_DIR} already contains checkpoints." >&2
  echo "       Use RESUME=1 to continue it, or move the directory aside first." >&2
  exit 3
fi

RESUME_FLAG=""
[[ "${RESUME}" == "1" ]] && RESUME_FLAG="--resume"

echo "======================================================================"
echo "SOURCE TRAINING --- Cityscapes (cityscapes_fine_mtl_train, 2975 images)"
echo "EXP    : ${EXP}"
echo "GPU    : ${GPU}"
echo "SEED   : ${SEED}"
echo "CONFIG : ${CFG}"
echo "RESUME : ${RESUME}"
echo "TF32   : ${TF32}"
echo "OUT    : ${HOST_OUT_DIR}"
echo "LOG    : ${LOG}"
echo "======================================================================"

docker run --rm \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NVIDIA_TF32_OVERRIDE="${TF32}" \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  -e FVCORE_CACHE=/workspace/output/.cache \
  -v "${HOST_REPO}:/workspace/amrod:ro" \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --num-gpus 1 \
      ${RESUME_FLAG} \
      OUTPUT_DIR ${OUT} \
      SEED ${SEED} ${EXTRA_OPTS}
  " 2>&1 | tee -a "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
echo "48k comparison checkpoint : ${HOST_OUT_DIR}/model_0047999.pth"
echo "60k final checkpoint      : ${HOST_OUT_DIR}/model_final.pth"
