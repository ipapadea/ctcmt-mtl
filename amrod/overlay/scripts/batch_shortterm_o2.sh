#!/usr/bin/env bash
set -euo pipefail

# Short-term protocols for the routing family, seed 0, on this machine.
#
#   Cityscapes-C short : 12 corruptions, single pass (12 evaluations)
#   ACDC short         :  4 conditions,  single pass ( 4 evaluations)
#
# Both existing short-term references (e11_bothsc_ctcrD_csc12 and _acdc4) run
# THRESHOLD_MAX 0.90, so these arms use the E11 lineage rather than the
# threshold-0.80 one built for the long Cityscapes-C stream. That is why the
# config filenames carry an `_acdc` suffix: it denotes the E11/thr-0.90
# lineage, NOT the dataset. The dataset comes from STREAM.
#
#   ctcmt_e25_detonly_acdc.yaml        = E11 + CTCMT_DET_ONLY        (ceiling)
#   ctcmt_e24_seghead_only_acdc.yaml   = E11 + aux_head_only         (S6, lambda=0)
#   ctcmt_e27_s6_entropy_acdc.yaml     = E24 + entropy-weighted CE   (O2)
#
# O2 alone would not be interpretable here: against the E11 short references it
# differs by routing AND entropy, so its two single-factor parents are included.
# Short runs are cheap -- the whole set is ~2.5 h over two GPUs.
#
#   [DRY_RUN=1] bash scripts/batch_shortterm_o2.sh [GPU_A] [GPU_B]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0 1)

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost
       fog brightness contrast elastic_transform pixelate jpeg_compression)
csc12_stream="("; for c in "${CSC12[@]}"; do csc12_stream+="\"${c}_mtl\","; done
csc12_stream="${csc12_stream%,})"
acdc4_stream='("acdc_fog_mtl","acdc_night_mtl","acdc_rain_mtl","acdc_snow_mtl")'

# proto:name:config:expected_evals
JOBS=(
  "csc12:e25_detonly_csc12_s0:${CFG}/ctcmt_e25_detonly_acdc.yaml:12"
  "csc12:e24_seghead_only_csc12_s0:${CFG}/ctcmt_e24_seghead_only_acdc.yaml:12"
  "csc12:e27_s6_entropy_csc12_s0:${CFG}/ctcmt_e27_s6_entropy_acdc.yaml:12"
  "acdc4:e25_detonly_acdc4_s0:${CFG}/ctcmt_e25_detonly_acdc.yaml:4"
  "acdc4:e24_seghead_only_acdc4_s0:${CFG}/ctcmt_e24_seghead_only_acdc.yaml:4"
  "acdc4:e27_s6_entropy_acdc4_s0:${CFG}/ctcmt_e27_s6_entropy_acdc.yaml:4"
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
  else TODO+=("${job}"); fi
done
[[ ${#TODO[@]} -eq 0 ]] && { echo "Nothing to do."; exit 0; }

echo "######################################################################"
echo "SHORT-TERM PROTOCOLS   seed 0   (${#TODO[@]} runs over ${#GPUS[@]} GPU(s))"
for i in "${!TODO[@]}"; do
  IFS=: read -r proto name _ want <<< "${TODO[$i]}"
  echo "  gpu ${GPUS[$((i % ${#GPUS[@]}))]}  ${name}  (${proto}, ${want} evals)"
done
echo "  references: e11 csc12 17.0 AP50 / 28.7 mIoU | e11 acdc4 38.4 / 35.8"
echo "######################################################################"
if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "DRY_RUN=1 -- nothing launched."; exit 0; fi

run_one () {
  IFS=: read -r proto name cfg _ <<< "$2"
  echo "[gpu $1] START ${name}  $(date +%H:%M:%S)"
  if [[ "${proto}" == csc12 ]]; then
    STREAM="${csc12_stream}" bash "${HERE}/run_mixed_lt_local.sh" "$1" "${name}" "${cfg}" 0
  else
    STREAM="${acdc4_stream}" bash "${HERE}/run_ctcr_acdc_local.sh" "$1" "${name}" "${cfg}" 0
  fi || echo "[gpu $1] !!! ${name} FAILED"
  echo "[gpu $1] DONE  ${name}  $(date +%H:%M:%S)"
}

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
echo "DONE. Refresh the short-term tables:"
echo "  python3 scripts/make_results_md.py"
