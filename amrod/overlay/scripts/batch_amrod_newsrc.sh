#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="detectron2/configs/Cityscapes"

# ------------------------------------------------------------
# 1. Cityscapes-C LT: 5 corruptions x 10
# AMROD is detection-only, so use plain dataset names.
# ------------------------------------------------------------
CYCLE_CSC=(fog motion_blur snow brightness defocus_blur)

CSC_STREAM="("
for _ in $(seq 1 10); do
    for c in "${CYCLE_CSC[@]}"; do
        CSC_STREAM+="\"${c}\","
    done
done
CSC_STREAM="${CSC_STREAM%,})"

echo "============================================================"
echo "AMROD + NEW SOURCE -- Cityscapes-C LT"
echo "============================================================"

STREAM="${CSC_STREAM}" bash "${HERE}/run_mixed_lt_local.sh" \
    "${GPU}" \
    "amrod_newsrc_cscLT_s0" \
    "${CFG}/amrod_newsrc_csc.yaml" \
    0

# ------------------------------------------------------------
# 2. ACDC LT: fog -> night -> rain -> snow x 10
# Again plain detection datasets, no _mtl suffix.
# ------------------------------------------------------------
CYCLE_ACDC=(fog night rain snow)

ACDC_STREAM="("
for _ in $(seq 1 10); do
    for c in "${CYCLE_ACDC[@]}"; do
        ACDC_STREAM+="\"acdc_${c}\","
    done
done
ACDC_STREAM="${ACDC_STREAM%,})"

echo "============================================================"
echo "AMROD + NEW SOURCE -- ACDC LT"
echo "============================================================"

STREAM="${ACDC_STREAM}" bash "${HERE}/run_ctcr_acdc_local.sh" \
    "${GPU}" \
    "amrod_newsrc_acdcLT_s0" \
    "${CFG}/amrod_newsrc_acdc.yaml" \
    0

echo "AMROD NEW-SOURCE BATCH DONE"
