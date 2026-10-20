#!/usr/bin/env bash
# Simple ACDC CTTA (1 round) for TriLiteNet ablation on GPU 3.
#
# Runs three variants sequentially:
#   1. ctcmt_det  — det distillation only  (seg/ctcl/proto disabled)
#   2. ctcmt_seg  — seg pseudo-labels only (det/ctcl/proto disabled)
#   3. ctcmt_mtl  — full CT-CMT (det+seg+ctcl+proto)
#
# Usage:  bash scripts/run_trilit_acdc_ablation.sh
# Logs:   /tmp/trilit_acdc_<variant>.log

set -euo pipefail

GPU=3
CKPT_DET="runs/cityscapes_mtl_base_native/ckpt_best_det.pth"
CKPT_SEG="runs/cityscapes_mtl_base_native/ckpt_best_seg.pth"

BASE_DOCKER="docker run --rm --gpus '\"device=${GPU}\"' --shm-size=8g \
  -v /home/ilias/TriLiteNet:/workspace \
  -v /data/ilias/acdc:/data/ilias/acdc:ro \
  trilitenet:latest"

BASE_ARGS="--num-seg-classes 19 --num-det-classes 8 --device 0 --num-repeats 1"

echo "=== TriLiteNet ACDC ablation (GPU ${GPU}, 1 round) ==="

# 1 — CT-CMT detection-only
echo "[1/3] ctcmt_det ..."
eval ${BASE_DOCKER} python tools/adapt_ctta.py \
  --ckpt "${CKPT_DET}" --method ctcmt ${BASE_ARGS} \
  --ctcmt-seg-weight 0.0 --ctcmt-ctcl-weight 0.0 \
  --no-ctcmt-proto \
  --run-name cityscapes_mtl_base_native__ctcmt_det__x1 \
  2>&1 | tee /tmp/trilit_acdc_ctcmt_det.log
echo "[1/3] done"

# 2 — CT-CMT segmentation-only
echo "[2/3] ctcmt_seg ..."
eval ${BASE_DOCKER} python tools/adapt_ctta.py \
  --ckpt "${CKPT_SEG}" --method ctcmt ${BASE_ARGS} \
  --ctcmt-det-weight 0.0 --ctcmt-ctcl-weight 0.0 \
  --no-ctcmt-proto \
  --run-name cityscapes_mtl_base_native__ctcmt_seg__x1 \
  2>&1 | tee /tmp/trilit_acdc_ctcmt_seg.log
echo "[2/3] done"

# 3 — CT-CMT full MTL
echo "[3/3] ctcmt_mtl ..."
eval ${BASE_DOCKER} python tools/adapt_ctta.py \
  --ckpt "${CKPT_DET}" --method ctcmt ${BASE_ARGS} \
  --run-name cityscapes_mtl_base_native__ctcmt_mtl__x1 \
  2>&1 | tee /tmp/trilit_acdc_ctcmt_mtl.log
echo "[3/3] done"

echo ""
echo "=== Results ==="
grep -h "global\|per-weather" \
  /tmp/trilit_acdc_ctcmt_det.log \
  /tmp/trilit_acdc_ctcmt_seg.log \
  /tmp/trilit_acdc_ctcmt_mtl.log 2>/dev/null || true
