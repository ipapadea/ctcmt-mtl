#!/usr/bin/env bash
set -euo pipefail
cd /home/ilias/AMROD

A_CFG="detectron2/configs/Cityscapes/ctcmt_e8a_v2_full_ctcr_no_ctpv.yaml"
B_CFG="detectron2/configs/Cityscapes/ctcmt_e8b_v2_hard_ctcr_no_ctpv.yaml"
C_CFG="detectron2/configs/Cityscapes/ctcmt_e8c_v2_soft_ctcr_no_ctpv.yaml"

(
  bash scripts/run_ctcr_one.sh 2 ctcr_A_v2_full_no_ctpv_fogx10 "$A_CFG" 0
  bash scripts/run_ctcr_one.sh 2 ctcr_B_v2_hard_t03_no_ctpv_fogx10 "$B_CFG" 0
) &
PID_GPU2=$!

(
  bash scripts/run_ctcr_one.sh 3 ctcr_C_v2_soft_a02_no_ctpv_fogx10 "$C_CFG" 0
) &
PID_GPU3=$!

echo "GPU2 worker PID: ${PID_GPU2} (A -> B)"
echo "GPU3 worker PID: ${PID_GPU3} (C)"

FAIL=0
wait "${PID_GPU2}" || FAIL=1
wait "${PID_GPU3}" || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one CT-CR ablation worker failed."
  exit 1
fi

echo "All A/B/C runs completed."
