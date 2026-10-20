#!/usr/bin/env bash
set -euo pipefail

# Same-source baselines on Cityscapes-C, both protocols, on ONE gpu.
#
# The Cityscapes-C analogue of run_acdc_lt_baselines_local.sh. Every arm loads
# the same panoptic_fpn_R50_cityscapes checkpoint as our own runs, so the
# comparison isolates the adaptation method.
#
#   cscLT  (fog -> motion -> snow -> bright -> defocus) x 10 = 50 evals
#   csc12  twelve corruptions, single pass              = 12 evals
#
# AMROD is detection-only and takes the plain `{corruption}` datasets (coco
# evaluator); TENT and CoTTA are segmentation-only and take `{corruption}_semseg`
# (sem_seg evaluator). Mixing those up yields KeyError on the missing metric.
#
# amrod_pfnsrc_cscLT_s0 already exists as amrod_pfnsrc_cscLT_cronus_s0 and is
# not re-run here. Runs that already have a full log are skipped, so this is
# safe to re-invoke.
#
#   bash scripts/run_csc_baselines_local.sh [GPU] [SEED]
#   DRY_RUN=1 bash scripts/run_csc_baselines_local.sh 1     # show the queue only

GPU="${1:-1}"
SEED="${2:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
DRY_RUN="${DRY_RUN:-0}"

LT_CYCLE=(fog motion_blur snow brightness defocus_blur)
SHORT=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness
       contrast elastic_transform pixelate jpeg_compression)

mk_stream() {  # mk_stream SUFFIX ROUNDS NAMES...
  local suffix="$1" rounds="$2"; shift 2
  local s="("
  for _ in $(seq 1 "${rounds}"); do
    for c in "$@"; do s+="\"${c}${suffix}\","; done
  done
  echo "${s%,})"
}

lt_det=$(mk_stream ""        10 "${LT_CYCLE[@]}")
lt_seg=$(mk_stream "_semseg" 10 "${LT_CYCLE[@]}")
s12_det=$(mk_stream ""        1 "${SHORT[@]}")
s12_seg=$(mk_stream "_semseg" 1 "${SHORT[@]}")

queue=()
consider() {  # consider EXP CONFIG STREAM EXPECTED_EVALS
  local log="${HOST_OUT}/logs/$1.log"
  local have=0
  [[ -f "${log}" ]] && have=$(grep -c "in csv format" "${log}" || true)
  if [[ "${have}" -ge "$4" ]]; then
    echo "  SKIP $1 (complete, ${have} evals)"
    return 0
  fi
  [[ "${have}" -gt 0 ]] && echo "  REDO $1 (only ${have}/$4 evals)"
  queue+=("$1|$2|$3")
}

# Short protocol first: it fills Tables 5-6, which currently have no baseline.
consider "amrod_pfnsrc_csc12_s${SEED}" "${CFG}/amrod_pfn_R_50_CS_C.yaml"         "${s12_det}" 12
consider "tent_pfnsrc_csc12_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_CS_C.yaml"   "${s12_seg}" 12
consider "cotta_pfnsrc_csc12_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_CS_C.yaml"  "${s12_seg}" 12
consider "tent_pfnsrc_cscLT_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_CS_C.yaml"   "${lt_seg}"  50
consider "cotta_pfnsrc_cscLT_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_CS_C.yaml"  "${lt_seg}"  50

echo "######################################################################"
echo "CITYSCAPES-C SAME-SOURCE BASELINES   gpu ${GPU}   seed ${SEED}"
if [[ "${#queue[@]}" -eq 0 ]]; then
  echo "  nothing to do"; echo "######################################################################"; exit 0
fi
for job in "${queue[@]}"; do echo "  ${job%%|*}"; done
echo "  reference: AMROD cscLT 26.40 AP50 | ours full MTL 24.80 | det-only 26.80"
echo "######################################################################"

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN=${DRY_RUN} -- nothing launched."; exit 0
fi

for job in "${queue[@]}"; do
  IFS="|" read -r exp cfg stream <<<"${job}"
  echo "=== ${exp}"
  STREAM="${stream}" bash "${HERE}/run_mixed_lt_local.sh" \
    "${GPU}" "${exp}" "${cfg}" "${SEED}" || echo "!!! ${exp} FAILED"
done

echo "CITYSCAPES-C BASELINE SUITE DONE (gpu ${GPU}, seed ${SEED})"
