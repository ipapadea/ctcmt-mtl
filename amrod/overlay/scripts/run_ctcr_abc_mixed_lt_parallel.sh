#!/usr/bin/env bash
set -euo pipefail

# Default two-GPU schedule:
#   GPU_A=2: A then B
#   GPU_C=3: C
#
# Override on Cronus, e.g.:
#   GPU_AB=0 GPU_C=1 \
#   HOST_REPO=/home/ilias/AMROD \
#   HOST_OUT=/data/... \
#   CSC_ROOT=/data/.../cityscapes_c_amrod \
#   CITYSCAPES_ROOT=/data/.../cityscapes_pfn \
#   CITYSCAPES_ANN_ROOT=/data/.../cityscapes/annotations \
#   bash scripts/run_ctcr_abc_mixed_lt_parallel.sh

GPU_AB="${GPU_AB:-2}"
GPU_C="${GPU_C:-3}"
SEED="${SEED:-0}"

cd "${HOST_REPO:-/home/ilias/AMROD}"

A_CFG="detectron2/configs/Cityscapes/ctcmt_e8a_v2_full_ctcr_no_ctpv.yaml"
B_CFG="detectron2/configs/Cityscapes/ctcmt_e8b_v2_hard_ctcr_no_ctpv.yaml"
C_CFG="detectron2/configs/Cityscapes/ctcmt_e8c_v2_soft_ctcr_no_ctpv.yaml"

(
  bash scripts/run_ctcr_mixed_lt_one.sh \
    "${GPU_AB}" \
    ctcr_A_v2_full_no_ctpv_csc_mixed_lt_x10 \
    "${A_CFG}" \
    "${SEED}"

  bash scripts/run_ctcr_mixed_lt_one.sh \
    "${GPU_AB}" \
    ctcr_B_v2_hard_t03_no_ctpv_csc_mixed_lt_x10 \
    "${B_CFG}" \
    "${SEED}"
) &
PID_AB=$!

(
  bash scripts/run_ctcr_mixed_lt_one.sh \
    "${GPU_C}" \
    ctcr_C_v2_soft_a02_no_ctpv_csc_mixed_lt_x10 \
    "${C_CFG}" \
    "${SEED}"
) &
PID_C=$!

echo "GPU ${GPU_AB}: A -> B (PID ${PID_AB})"
echo "GPU ${GPU_C}: C      (PID ${PID_C})"

FAIL=0
wait "${PID_AB}" || FAIL=1
wait "${PID_C}" || FAIL=1

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one mixed long-term worker failed." >&2
  exit 1
fi

echo "All mixed long-term A/B/C runs completed."
