#!/usr/bin/env bash
set -euo pipefail

# Table 5 (ACDC long-term x10) baselines, all on OUR Panoptic-FPN checkpoint and
# all on THIS machine.
#
# Run here rather than on gpu1 for two reasons:
#   1. gpu1 registers only acdc_fog_semseg, so CoTTA/TENT cannot run there.
#   2. our own ACDC x10 seeds are hardware-mixed; keeping every row of the table
#      on one GPU type removes that confound.
#
#   bash scripts/run_acdc_lt_baselines_local.sh GPU [SEED] [amrod|cotta|tent|all]

GPU="${1:?GPU id required}"
SEED="${2:-0}"
ONLY="${3:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes

CYCLE=(fog night rain snow)
det_stream="("; seg_stream="("
for _ in $(seq 1 10); do
  for c in "${CYCLE[@]}"; do
    det_stream+="\"acdc_${c}\","
    seg_stream+="\"acdc_${c}_semseg\","
  done
done
det_stream="${det_stream%,})"
seg_stream="${seg_stream%,})"

launch() {  # launch METHOD EXP CONFIG STREAM
  if [[ "${ONLY}" != "all" && "${ONLY}" != "$1" ]]; then
    echo "--- skip $2 (ONLY=${ONLY})"; return 0
  fi
  STREAM="$4" bash "${HERE}/run_ctcr_acdc_local.sh" "${GPU}" "$2" "$3" "${SEED}"
}

launch amrod "amrod_pfnsrc_acdcLT_s${SEED}" "${CFG}/amrod_pfn_R_50_ACDC.yaml"        "${det_stream}"
launch cotta "cotta_pfnsrc_acdcLT_s${SEED}" "${CFG}/cotta_semseg_pfn_R_50_ACDC.yaml" "${seg_stream}"
launch tent  "tent_pfnsrc_acdcLT_s${SEED}"  "${CFG}/tent_semseg_pfn_R_50_ACDC.yaml"  "${seg_stream}"

echo "ACDC-LT BASELINE SUITE DONE (seed ${SEED}, ONLY=${ONLY})"
