#!/usr/bin/env bash
set -euo pipefail

# Seeds 42 and 123 for one of the routing options, on both long-term protocols.
#
#   OPT=e27_s6_entropy       bash scripts/batch_seeds_options.sh 0 1 2 3   (default)
#   OPT=e26_partial025       bash scripts/batch_seeds_options.sh 0 1 2 3
#   OPT=e28_adaptive_routing bash scripts/batch_seeds_options.sh 0 1 2 3
#
# Four runs: {Cityscapes-C x10, ACDC x10} x {42, 123}. With four GPUs each run
# gets its own, so wall clock is one Cityscapes-C run (~6 h on a 3090, less on
# an L40S). With fewer GPUs, pass fewer ids and they queue up.
#
# Reference means at seed 0, for reading the result:
#   Cityscapes-C   E13a 25.07+-0.44 / 31.04+-0.27   S6 26.76+-0.14 / 34.60+-0.13
#                  E15 (ceiling) 26.93+-0.20 / 35.70+-0.16
#   ACDC           E11 43.81+-0.22 / 40.03+-0.27    E24 (S6) 43.46 / 37.64
#                  E25 (det-only) 43.38 / 38.85
#
# Already-complete runs are skipped, so re-running only fills what is missing.

OPT="${OPT:-e27_s6_entropy}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
export HOST_REPO="${HOST_REPO:-$HOME/AMROD}"
export HOST_OUT="${HOST_OUT:-/data/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"
export ACDC_ROOT="${ACDC_ROOT:-/data/ilias/acdc}"

GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0 1 2 3)

for suffix in "" _acdc; do
  [[ -f "${HERE}/../${CFG}/ctcmt_${OPT}${suffix}.yaml" ]] || {
    echo "ERROR: missing ${CFG}/ctcmt_${OPT}${suffix}.yaml -- git pull first" >&2; exit 2; }
done
[[ -f "${HOST_OUT}/panoptic_fpn_R50_cityscapes/model_final.pth" ]] || {
  echo "ERROR: missing source checkpoint under HOST_OUT=${HOST_OUT}" >&2; exit 2; }

csc_stream="("; for _ in $(seq 1 10); do
  for c in fog motion_blur snow brightness defocus_blur; do csc_stream+="\"${c}_mtl\","; done
done; csc_stream="${csc_stream%,})"
acdc_stream="("; for _ in $(seq 1 10); do
  for w in fog night rain snow; do acdc_stream+="\"acdc_${w}_mtl\","; done
done; acdc_stream="${acdc_stream%,})"

# Ordered so that round-robin over N GPUs gives each one a mix of the two
# protocols rather than all the long Cityscapes-C runs on a single GPU.
JOBS=()
for s in 42 123; do
  JOBS+=("csc:${OPT}_cscLT_s${s}:${CFG}/ctcmt_${OPT}.yaml:${s}:50")
done
for s in 42 123; do
  JOBS+=("acdc:${OPT}_acdc_acdcLT_s${s}:${CFG}/ctcmt_${OPT}_acdc.yaml:${s}:40")
done

TODO=()
for job in "${JOBS[@]}"; do
  IFS=: read -r _ name _ _ want <<< "${job}"
  log="${HOST_OUT}/logs/${name}.log"; n=0
  [[ -f "${log}" ]] && n="$(grep -c 'in csv format' "${log}" || true)"
  if [[ "${n}" -eq "${want}" ]]; then echo "  SKIP ${name} (complete, ${n} evals)"
  else TODO+=("${job}"); fi
done
[[ ${#TODO[@]} -eq 0 ]] && { echo "Nothing to do."; exit 0; }

echo "######################################################################"
echo "SEEDS 42/123 for ${OPT}   (${#TODO[@]} runs over ${#GPUS[@]} GPU(s))"
for i in "${!TODO[@]}"; do
  IFS=: read -r proto name _ seed _ <<< "${TODO[$i]}"
  echo "  gpu ${GPUS[$((i % ${#GPUS[@]}))]}  ${name}  (${proto}, seed ${seed})"
done
echo "######################################################################"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 -- nothing launched."
  exit 0
fi

run_one () {
  IFS=: read -r proto name cfg seed _ <<< "$2"
  echo "[gpu $1] START ${name}  $(date +%H:%M:%S)"
  if [[ "${proto}" == csc ]]; then
    STREAM="${csc_stream}" bash "${HERE}/run_mixed_lt_local.sh" "$1" "${name}" "${cfg}" "${seed}"
  else
    STREAM="${acdc_stream}" bash "${HERE}/run_ctcr_acdc_local.sh" "$1" "${name}" "${cfg}" "${seed}"
  fi || echo "[gpu $1] !!! ${name} FAILED"
  echo "[gpu $1] DONE  ${name}  $(date +%H:%M:%S)"
}

# One subshell per GPU, running that GPU's jobs strictly in sequence. Spawning
# a subshell per JOB would put every run on the GPUs at once.
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
echo "DONE. Aggregate:"
echo "  python3 scripts/report_screening_s1_s5.py --seeds 0,42,123 --protocol csc"
echo "  python3 scripts/report_screening_s1_s5.py --seeds 0,42,123 --protocol acdc"
