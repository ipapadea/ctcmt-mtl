#!/usr/bin/env bash
set -euo pipefail

# Detector-dominant cross-task batch (E34-E38), one sequential queue per GPU.
#
#   bash scripts/batch_detdominant.sh 1          # everything on GPU 1
#   DRY_RUN=1 bash scripts/batch_detdominant.sh 1
#   ONLY=e37 bash scripts/batch_detdominant.sh 1 # one arm
#
# Each arm is a SINGLE factor against the reference named beside it. Every arm
# already has CTPV off, inherited from e8a.
#
#   E34  CT-CR: highest-scoring box wins an overlap      vs E13a   cscLT
#   E35  CT-CR: weight each box by det_score             vs E13a   cscLT
#   E36  ... with gamma 3.0                              vs E35    cscLT
#   E37  detector vetoes the seg soft-CE (inverted CTPV) vs E13a   cscLT
#   E38  route only the FPN, not the whole trunk         vs E22/S6 cscLT
#   E38a same, on ACDC                                   vs E24    acdcLT
#
# Reference means to read the results against (seed 0 unless noted):
#   cscLT   E13a 25.07+-0.44 / 31.04+-0.27   (n=3, full MTL)
#           E22/S6 26.76+-0.14 / 34.60+-0.13 (n=3, blanket routing)
#           E15 detection ceiling 26.93+-0.20 / 35.70+-0.16 (n=3)
#   acdcLT  E11 43.81+-0.22 / 40.03+-0.27 (n=3)   E24/S6 43.46 / 37.64
#   Noise floor ~0.4 mAP0.5 on a single-seed difference: treat anything
#   smaller as unresolved.
#
# Completed runs are skipped, so re-running only fills what is missing.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
export HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
export HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
export ACDC_ROOT="${ACDC_ROOT:-/media/ilias/DATA/ilias/acdc}"

SEED="${SEED:-0}"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(1)

csc_stream="("; for _ in $(seq 1 10); do
  for c in fog motion_blur snow brightness defocus_blur; do csc_stream+="\"${c}_mtl\","; done
done; csc_stream="${csc_stream%,})"
acdc_stream="("; for _ in $(seq 1 10); do
  for w in fog night rain snow; do acdc_stream+="\"acdc_${w}_mtl\","; done
done; acdc_stream="${acdc_stream%,})"

# proto:name:config:expected_evals
JOBS=(
  "csc:e34_ctcr_scoreorder_cscLT_s${SEED}:${CFG}/ctcmt_e34_ctcr_scoreorder.yaml:50"
  "csc:e37_seg_detveto_cscLT_s${SEED}:${CFG}/ctcmt_e37_seg_detveto.yaml:50"
  "csc:e38_fpn_routing_cscLT_s${SEED}:${CFG}/ctcmt_e38_fpn_routing.yaml:50"
  "csc:e35_ctcr_detscore_cscLT_s${SEED}:${CFG}/ctcmt_e35_ctcr_detscore.yaml:50"
  "csc:e36_ctcr_detscore_g3_cscLT_s${SEED}:${CFG}/ctcmt_e36_ctcr_detscore_g3.yaml:50"
  "acdc:e38_fpn_routing_acdcLT_s${SEED}:${CFG}/ctcmt_e38_fpn_routing_acdc.yaml:40"
)

for job in "${JOBS[@]}"; do
  IFS=: read -r _ _ cfg _ <<< "${job}"
  [[ -f "${HERE}/../${cfg}" ]] || { echo "ERROR: missing ${cfg}" >&2; exit 2; }
done
[[ -f "${HOST_OUT}/panoptic_fpn_R50_cityscapes/model_final.pth" ]] || {
  echo "ERROR: missing source checkpoint under HOST_OUT=${HOST_OUT}" >&2; exit 2; }

TODO=()
for job in "${JOBS[@]}"; do
  IFS=: read -r _ name _ want <<< "${job}"
  log="${HOST_OUT}/logs/${name}.log"; n=0
  [[ -f "${log}" ]] && n="$(grep -c 'in csv format' "${log}" || true)"
  if [[ "${n}" -eq "${want}" ]]; then echo "  SKIP ${name} (complete, ${n} evals)"
  elif [[ -n "${ONLY:-}" && "${name}" != *"${ONLY}"* ]]; then echo "  SKIP ${name} (ONLY=${ONLY})"
  else TODO+=("${job}"); fi
done
[[ ${#TODO[@]} -eq 0 ]] && { echo "Nothing to do."; exit 0; }

echo "======================================================================"
echo "DETECTOR-DOMINANT BATCH  seed ${SEED}   (${#TODO[@]} runs over ${#GPUS[@]} GPU(s))"
for i in "${!TODO[@]}"; do
  IFS=: read -r proto name _ _ <<< "${TODO[$i]}"
  echo "  gpu ${GPUS[$((i % ${#GPUS[@]}))]}  ${name}  (${proto})"
done
echo "  est. $(( ${#TODO[@]} * 5 ))-$(( ${#TODO[@]} * 6 )) h total per GPU queue"
echo "======================================================================"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 -- nothing launched."
  exit 0
fi

run_one () {
  IFS=: read -r proto name cfg _ <<< "$2"
  echo "[gpu $1] START ${name}  $(date +%H:%M:%S)"
  if [[ "${proto}" == csc ]]; then
    STREAM="${csc_stream}" bash "${HERE}/run_mixed_lt_local.sh" "$1" "${name}" "${cfg}" "${SEED}"
  else
    STREAM="${acdc_stream}" bash "${HERE}/run_ctcr_acdc_local.sh" "$1" "${name}" "${cfg}" "${SEED}"
  fi || echo "[gpu $1] !!! ${name} FAILED"
  echo "[gpu $1] DONE  ${name}  $(date +%H:%M:%S)"
}

# One subshell per GPU, running that GPU's jobs strictly in sequence.
pids=()
for g in "${!GPUS[@]}"; do
  (
    for i in "${!TODO[@]}"; do
      [[ $((i % ${#GPUS[@]})) -eq ${g} ]] || continue
      run_one "${GPUS[$g]}" "${TODO[$i]}"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done

echo
echo "DONE. Check completeness, then regenerate results:"
echo "  for f in ${HOST_OUT}/logs/e3{4,5,6,7,8}*_s${SEED}.log; do"
echo "    echo \"\$(grep -c 'in csv format' \$f) \$(grep -c Traceback \$f) \$f\"; done"
echo "  python3 scripts/make_results_md.py"
