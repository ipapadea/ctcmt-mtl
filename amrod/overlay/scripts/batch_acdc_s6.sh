#!/usr/bin/env bash
set -euo pipefail

# S6 (seg-head-only gradient routing) on ACDC x10, with the det-only control
# that makes it interpretable.
#
#   (acdc_fog -> night -> rain -> snow) x 10 = 40 evaluations, no reset.
#
#   E24  e11 + CTCMT_CONFLICT_MODE=aux_head_only   (S6 on ACDC)
#   E25  e11 + CTCMT_DET_ONLY                      (the ACDC analogue of E15)
#
# Both are ONE factor from E11, the reported ACDC recipe. THRESHOLD_MAX stays
# 0.90 here: the 0.80 ceiling was a Cityscapes-C-specific fix.
#
# Reference, seed 0, same source checkpoint:
#            AP50    mIoU
#   e11      43.98   39.80   full MTL  <- the ACDC method
#   AMROD    39.00     --
#   TENT       --     32.02
#   CoTTA      --     20.17 (collapses 30.9 -> 14.0)
#
# How to read it, once E25 lands:
#   E25 > E11  -> ACDC has negative transfer too; E24 should close the gap
#   E25 ~ E11  -> no negative transfer on ACDC; E24 should be a no-op
#   E25 < E11  -> segmentation HELPS detection on ACDC, and E24 should hurt
#
#   bash scripts/batch_acdc_s6.sh [GPU_A] [GPU_B]

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

CYCLE=(fog night rain snow)
stream="("
for _ in $(seq 1 10); do
  for w in "${CYCLE[@]}"; do stream+="\"acdc_${w}_mtl\","; done
done
stream="${stream%,})"

# Det-only still evaluates mIoU: the teacher keeps its seg head, and on CS-C
# that is exactly how E15's 35.51 mIoU was measured. Use the MTL stream for both.
echo "######################################################################"
echo "S6 ON ACDC   seed 0   (fog -> night -> rain -> snow) x 10 = 40 evals"
echo "  gpu ${GPU_A}: e24_seghead_only_acdc_acdcLT_s0   (aux_head_only routing)"
echo "  gpu ${GPU_B}: e25_detonly_acdc_acdcLT_s0        (det-only control)"
echo "  reference: e11 43.98 AP50 / 39.80 mIoU | AMROD 39.00"
echo "######################################################################"

STREAM="${stream}" bash "${HERE}/run_ctcr_acdc_local.sh" \
  "${GPU_A}" e24_seghead_only_acdc_acdcLT_s0 \
  "${CFG}/ctcmt_e24_seghead_only_acdc.yaml" 0 & pid_a=$!
STREAM="${stream}" bash "${HERE}/run_ctcr_acdc_local.sh" \
  "${GPU_B}" e25_detonly_acdc_acdcLT_s0 \
  "${CFG}/ctcmt_e25_detonly_acdc.yaml" 0 & pid_b=$!
wait "${pid_a}" || echo "!!! e24 failed"
wait "${pid_b}" || echo "!!! e25 failed"

echo
echo "DONE. Both must show 40 evals, 0 tracebacks, and e24 applied_rate=1.0000:"
echo "  python3 scripts/report_screening_s1_s5.py --protocol acdc"
