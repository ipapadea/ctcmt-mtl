#!/usr/bin/env bash
set -euo pipefail

# Follow-up to the S1..S5 screening: the two arms that test what the factorial
# ablation actually implicated.
#
#   E22  seg adapts its own head only, detection owns the shared trunk
#        (CTCMT_CONFLICT_MODE=aux_head_only, unconditional routing)
#   E23  clean DET x SEG factorial control: det + seg soft-CE both flow through
#        the trunk normally, cross-task losses (CT-CL, CT-CR, proto) all OFF
#
# One arm per GPU, in parallel. ~6 h.
#
# Reference on this stream, same source checkpoint, seed 0:
#            AP50    mIoU
#   e13a     24.80   30.83   full MTL  (cross-task ON)
#   e15      26.80   35.51   det-only
#   e21      15.10   28.21   seg-only
#   source   13.17   27.21   no adaptation
#   AMROD    26.40     --
#
# Reading guide:
#   E22 -> AP50 ~26.8 AND mIoU >=36 would clear both targets at once.
#   E23 ~ e13a  -> cross-task losses innocent, det/seg interaction is the cause.
#   E23 ~ e15   -> CT-CL/CT-CR carry the damage, seg soft-CE is innocent.
#
#   bash scripts/batch_e22_e23.sh [GPU_A] [GPU_B]

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

echo "######################################################################"
echo "E22 / E23   seed 0   CS-C x10 (50 evals, no reset)"
echo "  gpu ${GPU_A}: e22_seghead_only_cscLT_s0   (aux_head_only routing)"
echo "  gpu ${GPU_B}: e23_detseg_nocross_cscLT_s0 (det+seg, no cross-task)"
echo "######################################################################"

bash "${HERE}/run_mixed_lt_local.sh" "${GPU_A}" e22_seghead_only_cscLT_s0 \
  "${CFG}/ctcmt_e22_seghead_only.yaml" 0 & pid_a=$!
bash "${HERE}/run_mixed_lt_local.sh" "${GPU_B}" e23_detseg_nocross_cscLT_s0 \
  "${CFG}/ctcmt_e23_detseg_nocross.yaml" 0 & pid_b=$!
wait "${pid_a}" || echo "!!! e22 failed"
wait "${pid_b}" || echo "!!! e23 failed"

echo
echo "DONE. Both must show 50 evals, 0 tracebacks, fallbacks=0:"
echo "  python3 scripts/report_screening_s1_s5.py --logs ${HOST_OUT}/logs"
