#!/usr/bin/env bash
set -euo pipefail

# Re-run of the four screening arms killed by the degenerate-CT-CL crash on
# 2026-09-12 (see scripts/test_conflict_modes.py case 7). S4/e19 completed all
# 50 evaluations and is NOT re-run: it never enters the per-component
# diagnostic path, and the fix does not touch its code path.
#
# Balanced two-per-GPU this time, unlike the 3/2 split that left gpu 0 idle.
#
#   bash scripts/batch_screening_rerun.sh [GPU_A] [GPU_B]

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"

QUEUE_A=(
  "e16_protectedgrad_cscLT_s0:${CFG}/ctcmt_e16_protectedgrad.yaml"
  "e18_harddecouple_cscLT_s0:${CFG}/ctcmt_e18_harddecouple.yaml"
)
QUEUE_B=(
  "e17_cagrad_cscLT_s0:${CFG}/ctcmt_e17_cagrad.yaml"
  "e20_dynweight_cscLT_s0:${CFG}/ctcmt_e20_dynweight.yaml"
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
echo "SCREENING RE-RUN (post grad-surgery crash fix)   seed 0   CS-C x10"
echo "  gpu ${GPU_A}: ${QUEUE_A[*]%%:*}"
echo "  gpu ${GPU_B}: ${QUEUE_B[*]%%:*}"
echo "  e19_frozentrunk already complete (50/50), not re-run"
echo "######################################################################"

run_queue "${GPU_A}" "${QUEUE_A[@]}" & pid_a=$!
run_queue "${GPU_B}" "${QUEUE_B[@]}" & pid_b=$!
wait "${pid_a}" || true
wait "${pid_b}" || true

echo
echo "RE-RUN DONE. Every arm must show 50 evals and fallbacks=0:"
echo "  python3 scripts/report_screening_s1_s5.py --logs ${HOST_OUT}/logs"
