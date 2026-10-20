#!/usr/bin/env bash
set -euo pipefail

# Sequential A -> B -> C mixed long-term run on one GPU.
# Usage:
#   bash scripts/run_ctcr_abc_mixed_lt_batch.sh [GPU]
#
# Path variables can be overridden in the environment exactly as in
# run_ctcr_mixed_lt_one.sh.

GPU="${1:-3}"
SEED="${SEED:-0}"

cd "${HOST_REPO:-/home/ilias/AMROD}"

bash scripts/run_ctcr_mixed_lt_one.sh \
  "${GPU}" \
  ctcr_A_v2_full_no_ctpv_csc_mixed_lt_x10 \
  detectron2/configs/Cityscapes/ctcmt_e8a_v2_full_ctcr_no_ctpv.yaml \
  "${SEED}"

bash scripts/run_ctcr_mixed_lt_one.sh \
  "${GPU}" \
  ctcr_B_v2_hard_t03_no_ctpv_csc_mixed_lt_x10 \
  detectron2/configs/Cityscapes/ctcmt_e8b_v2_hard_ctcr_no_ctpv.yaml \
  "${SEED}"

bash scripts/run_ctcr_mixed_lt_one.sh \
  "${GPU}" \
  ctcr_C_v2_soft_a02_no_ctpv_csc_mixed_lt_x10 \
  detectron2/configs/Cityscapes/ctcmt_e8c_v2_soft_ctcr_no_ctpv.yaml \
  "${SEED}"

echo "Sequential mixed long-term A/B/C batch completed on GPU ${GPU}."
