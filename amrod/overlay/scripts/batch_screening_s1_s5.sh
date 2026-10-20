#!/usr/bin/env bash
set -euo pipefail

# Seed-0 screening batch for detection/segmentation negative transfer on the
# Cityscapes-C long-term stream: (fog -> motion_blur -> snow -> brightness ->
# defocus_blur) x 10, no reset, 50 evaluations.
#
#   S1 e16  protected gradient projection   (asymmetric, detection protected)
#   S2 e17  CAGrad consensus                (symmetric literature control)
#   S3 e18  conflict-triggered decoupling   (discrete counterpart of S1)
#   S4 e19  frozen shared trunk             (structural isolation)
#   S5 e20  conflict-aware dynamic weight   (loss-level, cheapest)
#
# References already measured on this stream, same source checkpoint:
#   B0  e13a full MTL thr 0.8   AP50 24.80   (negative-transfer reference)
#   B1  e15  det-only thr 0.8   AP50 26.80   (detection control / ceiling)
#   AMROD on the PFN checkpoint AP50 26.26 / 26.40
# Target: AP50 -> ~26.8 while mIoU stays at or above e13a.
#
# Two sequential queues, one per GPU, run in parallel.
#
#   bash scripts/batch_screening_s1_s5.sh [GPU_A] [GPU_B]

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

# The gradient-surgery arms pay for an extra backward pass, so they are split
# across both queues rather than stacked on one.
QUEUE_A=(
  "e16_protectedgrad_cscLT_s0:${CFG}/ctcmt_e16_protectedgrad.yaml"
  "e18_harddecouple_cscLT_s0:${CFG}/ctcmt_e18_harddecouple.yaml"
)
QUEUE_B=(
  "e17_cagrad_cscLT_s0:${CFG}/ctcmt_e17_cagrad.yaml"
  "e20_dynweight_cscLT_s0:${CFG}/ctcmt_e20_dynweight.yaml"
  "e19_frozentrunk_cscLT_s0:${CFG}/ctcmt_e19_frozentrunk.yaml"
)

run_queue() {
  local gpu="$1"; shift
  for job in "$@"; do
    IFS=: read -r name cfg <<< "${job}"
    echo "[gpu ${gpu}] START ${name}  $(date +%H:%M:%S)"
    bash "${HERE}/run_mixed_lt_local.sh" "${gpu}" "${name}" "${cfg}" 0 \
      || echo "[gpu ${gpu}] !!! ${name} FAILED, continuing"
    echo "[gpu ${gpu}] DONE  ${name}  $(date +%H:%M:%S)"
  done
}

echo "######################################################################"
echo "NEGATIVE-TRANSFER SCREENING BATCH   seed 0   CS-C x10 (50 evals)"
echo "  gpu ${GPU_A}: ${QUEUE_A[*]%%:*}"
echo "  gpu ${GPU_B}: ${QUEUE_B[*]%%:*}"
echo "  refs: e13a 24.80 | e15 26.80 | AMROD 26.26"
echo "######################################################################"

run_queue "${GPU_A}" "${QUEUE_A[@]}" & pid_a=$!
run_queue "${GPU_B}" "${QUEUE_B[@]}" & pid_b=$!

# kill -0 on the recorded pids; pgrep -f would match this script itself.
wait "${pid_a}" || true
wait "${pid_b}" || true

echo
echo "BATCH DONE. Gather and compare:"
echo "  python3 scripts/gather_results.py --logs ${HOST_OUT}/logs"
echo "  python3 scripts/report_screening_s1_s5.py --logs ${HOST_OUT}/logs"
