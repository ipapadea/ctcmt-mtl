#!/usr/bin/env bash
set -euo pipefail

# Seeds 42 and 123 for the two Cityscapes-C long-term arms that are tied at
# n=1, so the tie can actually be claimed:
#
#   S6  e22 seg-head-only routing   AP50 26.74   mIoU 34.75
#   B1  e15 det-only                AP50 26.80   mIoU 35.51
#
# The measured same-seed noise floor is ~0.5 AP50 on the 50-eval mean (from the
# crashed/re-run replicate pair), so the 0.06 AP50 gap is unresolvable at one
# seed. mIoU is far tighter (~0.02), which makes the -0.76 mIoU gap the number
# most likely to survive replication.
#
# One e22 (6.2 h) + one e15 (3.1 h) per GPU => ~9.3 h wall clock.
#
# Optional: WITH_E13A=1 adds seeds 42/123 for the full-MTL reference, which
# also only has seed 0 on this protocol. Without it, every "vs e13a" margin
# stays an n=1 comparison. Adds ~5 h per GPU.
#
#   bash scripts/batch_seeds_s6_e15.sh [GPU_A] [GPU_B]

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
WITH_E13A="${WITH_E13A:-0}"

E22="${CFG}/ctcmt_e22_seghead_only.yaml"
E15="${CFG}/ctcmt_e15_detonly_thr080.yaml"
E13A="${CFG}/ctcmt_e13a_thrmax080.yaml"

# name:config:seed, longest job first so the two queues finish together.
QUEUE_A=("e22_seghead_only_cscLT_s42:${E22}:42" "e15_detonly_thr080_cscLT_s42:${E15}:42")
QUEUE_B=("e22_seghead_only_cscLT_s123:${E22}:123" "e15_detonly_thr080_cscLT_s123:${E15}:123")
if [[ "${WITH_E13A}" == "1" ]]; then
  QUEUE_A+=("e13a_thrmax080_cscLT_s42:${E13A}:42")
  QUEUE_B+=("e13a_thrmax080_cscLT_s123:${E13A}:123")
fi

# Skip jobs that already finished rather than aborting: run_mixed_lt_local.sh
# rm -rf's its output dir and truncates its log on start, so a completed run
# must never be re-entered, but a partly-written one is safe to redo. This
# makes the script idempotent - re-running it only fills what is missing.
filter_queue () {
  local job name log n
  for job in "$@"; do
    name="${job%%:*}"
    log="${HOST_OUT}/logs/${name}.log"
    n=0
    [[ -f "${log}" ]] && n="$(grep -c 'in csv format' "${log}" || true)"
    if [[ "${n}" -eq 50 ]]; then
      echo "  SKIP ${name} (already complete, 50 evals)" >&2
    else
      [[ "${n}" -gt 0 ]] && echo "  REDO ${name} (incomplete, ${n}/50 evals)" >&2
      echo "${job}"
    fi
  done
}
mapfile -t QUEUE_A < <(filter_queue "${QUEUE_A[@]}")
mapfile -t QUEUE_B < <(filter_queue "${QUEUE_B[@]}")
if [[ ${#QUEUE_A[@]} -eq 0 && ${#QUEUE_B[@]} -eq 0 ]]; then
  echo "Nothing to do: every requested run is already complete." >&2
  exit 0
fi

run_queue() {
  local gpu="$1"; shift
  [[ $# -eq 0 ]] && return 0
  for job in "$@"; do
    IFS=: read -r name cfg seed <<< "${job}"
    echo "[gpu ${gpu}] START ${name}  seed=${seed}  $(date +%H:%M:%S)"
    bash "${HERE}/run_mixed_lt_local.sh" "${gpu}" "${name}" "${cfg}" "${seed}" \
      || echo "[gpu ${gpu}] !!! ${name} FAILED, continuing"
    echo "[gpu ${gpu}] DONE  ${name}  $(date +%H:%M:%S)"
  done
}

echo "######################################################################"
echo "SEED REPLICATION   CS-C x10 (50 evals)   seeds 42 + 123"
echo "  gpu ${GPU_A}: ${QUEUE_A[*]%%:*}"
echo "  gpu ${GPU_B}: ${QUEUE_B[*]%%:*}"
echo "  seed-0 reference: e22 26.74/34.75 | e15 26.80/35.51 | e13a 24.80/30.83"
echo "  noise floor: ~0.5 AP50, ~0.02 mIoU on the 50-eval mean"
echo "######################################################################"

run_queue "${GPU_A}" "${QUEUE_A[@]}" & pid_a=$!
run_queue "${GPU_B}" "${QUEUE_B[@]}" & pid_b=$!
wait "${pid_a}" || true
wait "${pid_b}" || true

echo
echo "DONE. Aggregate across seeds:"
echo "  python3 scripts/report_screening_s1_s5.py --seeds 0,42,123"
