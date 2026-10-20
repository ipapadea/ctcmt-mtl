#!/usr/bin/env bash
set -euo pipefail

# Fisher-restoration diagnostic batch: two GPUs, two protocols, seed 0.
#
#   gpu A (Cityscapes-C, 50 evals each)   gpu B (ACDC, 40 evals each)
#     E30 = E13a + Fisher restore           E32 = E11  + Fisher restore
#     E31 = E22 (S6) + Fisher restore       E33 = E24 (S6) + Fisher restore
#
# Each arm is ONE factor from its reference. Every arm loads the same
# panoptic_fpn_R50_cityscapes checkpoint as the reference it is compared to.
#
# DIAGNOSTIC ONLY. CTCMT_FISHER_RESTORE is a port of AMROD's Randomized
# Restoration (Wei et al.). These rows quantify how much of the segmentation
# fix comes from that rule; they must never be reported as our contribution.
# The reportable mechanism is the rho-parameterised multi-task Fisher
#   F_i(rho) = g_det^2 + g_seg^2 + 2*rho*g_det*g_seg
# which is only worth building if these four runs show the effect is real.
#
# What we expect, and what would falsify it:
#   E30 > E13a on mIoU with AP50 flat  -> drift fix survives the current recipe
#   E31 ~ E15 on BOTH metrics          -> routing and restoration are complementary
#   E32 gain << E30 gain               -> consistent with a drift fix (ACDC drifts less)
#   E32 gain ~ E30 gain                -> NOT a drift fix; the explanation is wrong
#
# References, seed 0, same checkpoint (AP50 / mIoU):
#   cscLT   E13a 24.80/30.83   E22 S6 26.76/34.60   E15 ceiling 26.93/35.70
#           E10  23.99/31.00   E12 = E10+Fisher 24.00/32.74  (drift -5.09 -> -2.19)
#   acdcLT  E11  43.98/39.80   E24 S6 43.46/37.64   E25 det-only 43.38/38.85
#
#   bash scripts/batch_fisher_diag.sh [GPU_A] [GPU_B]
#   DRY_RUN=1 bash scripts/batch_fisher_diag.sh 0 1

GPU_A="${1:-0}"
GPU_B="${2:-1}"
SEED="${3:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
DRY_RUN="${DRY_RUN:-0}"

csc_stream="("; for _ in $(seq 1 10); do
  for c in fog motion_blur snow brightness defocus_blur; do csc_stream+="\"${c}_mtl\","; done
done; csc_stream="${csc_stream%,})"

acdc_stream="("; for _ in $(seq 1 10); do
  for w in fog night rain snow; do acdc_stream+="\"acdc_${w}_mtl\","; done
done; acdc_stream="${acdc_stream%,})"

# queue entries: EXP|CONFIG|RUNNER|STREAM|EXPECTED_EVALS
queue_a=(); queue_b=()
consider() {  # consider QUEUE_NAME EXP CONFIG RUNNER STREAM EVALS
  local -n q="$1"; shift
  local log="${HOST_OUT}/logs/$1.log" have=0
  [[ -f "${log}" ]] && have=$(grep -c "in csv format" "${log}" || true)
  if [[ "${have}" -ge "$5" ]]; then echo "  SKIP $1 (complete, ${have} evals)"; return 0; fi
  [[ "${have}" -gt 0 ]] && echo "  REDO $1 (only ${have}/$5)"
  q+=("$1|$2|$3|$4|$5")
}

consider queue_a "e30_fisher_full_cscLT_s${SEED}" "${CFG}/ctcmt_e30_fisher_full_csc.yaml" \
  run_mixed_lt_local.sh "${csc_stream}" 50
consider queue_a "e31_fisher_s6_cscLT_s${SEED}"   "${CFG}/ctcmt_e31_fisher_s6_csc.yaml" \
  run_mixed_lt_local.sh "${csc_stream}" 50
consider queue_b "e32_fisher_full_acdcLT_s${SEED}" "${CFG}/ctcmt_e32_fisher_full_acdc.yaml" \
  run_ctcr_acdc_local.sh "${acdc_stream}" 40
consider queue_b "e33_fisher_s6_acdcLT_s${SEED}"   "${CFG}/ctcmt_e33_fisher_s6_acdc.yaml" \
  run_ctcr_acdc_local.sh "${acdc_stream}" 40

echo "######################################################################"
echo "FISHER-RESTORATION DIAGNOSTIC BATCH   seed ${SEED}"
echo "  gpu ${GPU_A} (Cityscapes-C):"; for j in "${queue_a[@]:-}"; do [[ -n "$j" ]] && echo "    ${j%%|*}"; done
echo "  gpu ${GPU_B} (ACDC):";         for j in "${queue_b[@]:-}"; do [[ -n "$j" ]] && echo "    ${j%%|*}"; done
echo "  DIAGNOSTIC ONLY - AMROD's restoration rule, never reported as ours."
echo "######################################################################"

if [[ "${#queue_a[@]}" -eq 0 && "${#queue_b[@]}" -eq 0 ]]; then echo "nothing to do"; exit 0; fi
if [[ "${DRY_RUN}" != "0" ]]; then echo "DRY_RUN=${DRY_RUN} -- nothing launched."; exit 0; fi

run_queue() {  # run_queue GPU  (entries passed on stdin, one per line)
  local gpu="$1" job
  while IFS= read -r job; do
    [[ -z "${job}" ]] && continue
    IFS="|" read -r exp cfg runner stream _ <<<"${job}"
    echo "=== gpu ${gpu}: ${exp}"
    STREAM="${stream}" bash "${HERE}/${runner}" "${gpu}" "${exp}" "${cfg}" "${SEED}" \
      || echo "!!! ${exp} FAILED"
  done
}

printf '%s\n' "${queue_a[@]:-}" | run_queue "${GPU_A}" & pid_a=$!
printf '%s\n' "${queue_b[@]:-}" | run_queue "${GPU_B}" & pid_b=$!
wait "${pid_a}" || echo "!!! gpu ${GPU_A} queue failed"
wait "${pid_b}" || echo "!!! gpu ${GPU_B} queue failed"

echo
echo "FISHER DIAGNOSTIC BATCH DONE (seed ${SEED})"
echo "Each log must show its full eval count and 0 tracebacks before use."
