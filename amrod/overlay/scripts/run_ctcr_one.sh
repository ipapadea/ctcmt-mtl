#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="/home/ilias/AMROD"
HOST_OUT="/data/ilias/panoptic_fpn/output"
CSC_ROOT="/data/vgcmt/datasets/cityscapes_c_amrod"

OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="/home/ilias/${EXP}.log"

STREAM='("fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl","fog_mtl")'

echo "============================================================"
echo "CT-CR ABLATION"
echo "EXP    : ${EXP}"
echo "GPU    : ${GPU}"
echo "SEED   : ${SEED}"
echo "CONFIG : ${CFG}"
echo "STREAM : Cityscapes-C fog_mtl x10, continual/no reset"
echo "TF32   : OFF"
echo "============================================================"

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
  -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
  -v /data/vgcmt/datasets/cityscapes/annotations:/datasets/cityscapes/annotations:ro \
  -v "${CSC_ROOT}/fog:/datasets/fog:ro" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  amrod:latest bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only \
      --num-gpus 1 \
      OUTPUT_DIR ${OUT} \
      SEED ${SEED} \
      DATASETS.TEST '${STREAM}'
  " 2>&1 | tee "${LOG}"

echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
echo "OUT : ${HOST_OUT}/ctta_acdc/${EXP}"
