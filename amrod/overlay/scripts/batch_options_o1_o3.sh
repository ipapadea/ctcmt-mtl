#!/usr/bin/env bash
set -euo pipefail

# Options O1-O3: how much of the auxiliary gradient should reach the shared
# trunk, and when. Seed 0, both long-term protocols, six runs.
#
#   O1 e26  fixed lambda = 0.25
#   O2 e27  S6 (lambda = 0) + entropy-weighted semantic CE
#   O3 e28  lambda = 0.25 gated on teacher/anchor agreement (0 when drifting)
#
# All three sit on the S6 routing path, so lambda is the single knob:
#   lambda = 0     S6            (E22 on Cityscapes-C, E24 on ACDC)
#   lambda = 0.25  these arms
#   lambda = 1     plain joint   (E13a on Cityscapes-C, E11 on ACDC)
#
# Why both protocols: the sign of the transfer is opposite on the two, so the
# lambda curve should slope in opposite directions. O3 is the arm that could
# reconcile them with one rule.
#
#   Cityscapes-C x10 (50 evals)        AP50    mIoU
#     E13a lambda=1                   24.80   30.83
#     E22  lambda=0  (S6)             26.74   34.75
#     E15  det-only ceiling           26.80   35.51
#   ACDC x10 (40 evals)
#     E11  lambda=1                   43.98   39.80
#     E24  lambda=0  (S6)             43.46   37.64
#     E25  det-only                   43.38   38.85
#
# Three GPUs, one Cityscapes-C run then one ACDC run each.
#
#   bash scripts/batch_options_o1_o3.sh [GPU_A] [GPU_B] [GPU_C]
#
# Defaults below are the gpu1 paths; override per machine.

GPU_A="${1:-0}"; GPU_B="${2:-1}"; GPU_C="${3:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
export HOST_REPO="${HOST_REPO:-$HOME/AMROD}"
export HOST_OUT="${HOST_OUT:-/data/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"
export ACDC_ROOT="${ACDC_ROOT:-/data/ilias/acdc}"

csc_stream="("; for _ in $(seq 1 10); do
  for c in fog motion_blur snow brightness defocus_blur; do csc_stream+="\"${c}_mtl\","; done
done; csc_stream="${csc_stream%,})"
acdc_stream="("; for _ in $(seq 1 10); do
  for w in fog night rain snow; do acdc_stream+="\"acdc_${w}_mtl\","; done
done; acdc_stream="${acdc_stream%,})"

[[ -f "${HOST_OUT}/panoptic_fpn_R50_cityscapes/model_final.pth" ]] || {
  echo "ERROR: missing source checkpoint under HOST_OUT=${HOST_OUT}" >&2; exit 2; }
for c in e26_partial025 e27_s6_entropy e28_adaptive_routing; do
  for suffix in "" _acdc; do
    [[ -f "${HERE}/../${CFG}/ctcmt_${c}${suffix}.yaml" ]] || {
      echo "ERROR: missing config ctcmt_${c}${suffix}.yaml -- git pull first" >&2; exit 2; }
  done
done
echo "preflight OK: checkpoint and all six configs present"

run_pair () {   # gpu, option stem
  local gpu="$1" opt="$2"
  echo "[gpu ${gpu}] START ${opt} Cityscapes-C  $(date +%H:%M:%S)"
  STREAM="${csc_stream}" bash "${HERE}/run_mixed_lt_local.sh" \
    "${gpu}" "${opt}_cscLT_s0" "${CFG}/ctcmt_${opt}.yaml" 0 \
    || echo "[gpu ${gpu}] !!! ${opt} Cityscapes-C FAILED"
  echo "[gpu ${gpu}] START ${opt} ACDC  $(date +%H:%M:%S)"
  STREAM="${acdc_stream}" bash "${HERE}/run_ctcr_acdc_local.sh" \
    "${gpu}" "${opt}_acdc_acdcLT_s0" "${CFG}/ctcmt_${opt}_acdc.yaml" 0 \
    || echo "[gpu ${gpu}] !!! ${opt} ACDC FAILED"
  echo "[gpu ${gpu}] DONE  ${opt}  $(date +%H:%M:%S)"
}

echo "######################################################################"
echo "OPTIONS O1-O3   seed 0   Cityscapes-C x10 + ACDC x10"
echo "  gpu ${GPU_A}: e26_partial025       lambda=0.25"
echo "  gpu ${GPU_B}: e27_s6_entropy       lambda=0 + entropy-weighted CE"
echo "  gpu ${GPU_C}: e28_adaptive_routing lambda=0.25 gated on seg reliability"
echo "######################################################################"

run_pair "${GPU_A}" e26_partial025       & pid_a=$!
run_pair "${GPU_B}" e27_s6_entropy       & pid_b=$!
run_pair "${GPU_C}" e28_adaptive_routing & pid_c=$!
wait "${pid_a}" || true; wait "${pid_b}" || true; wait "${pid_c}" || true

echo
echo "DONE. Each run must show the right eval count, 0 tracebacks, and for the"
echo "routed arms a 'lam=' field in the periodic line (e28 also logs 'agree='):"
echo "  grep -o 'lam=[0-9.]*' ${HOST_OUT}/logs/e28_adaptive_routing_cscLT_s0.log | sort | uniq -c"
echo "  python3 scripts/report_screening_s1_s5.py --protocol csc"
echo "  python3 scripts/report_screening_s1_s5.py --protocol acdc"
