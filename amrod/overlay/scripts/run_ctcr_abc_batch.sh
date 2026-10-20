#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-3}"
cd /home/ilias/AMROD

bash scripts/run_ctcr_one.sh "$GPU" ctcr_A_v2_full_no_ctpv_fogx10 \
  detectron2/configs/Cityscapes/ctcmt_e8a_v2_full_ctcr_no_ctpv.yaml 0

bash scripts/run_ctcr_one.sh "$GPU" ctcr_B_v2_hard_t03_no_ctpv_fogx10 \
  detectron2/configs/Cityscapes/ctcmt_e8b_v2_hard_ctcr_no_ctpv.yaml 0

bash scripts/run_ctcr_one.sh "$GPU" ctcr_C_v2_soft_a02_no_ctpv_fogx10 \
  detectron2/configs/Cityscapes/ctcmt_e8c_v2_soft_ctcr_no_ctpv.yaml 0

echo "Sequential A/B/C batch completed on GPU ${GPU}."
