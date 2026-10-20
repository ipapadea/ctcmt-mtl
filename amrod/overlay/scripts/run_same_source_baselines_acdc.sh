#!/usr/bin/env bash
set -euo pipefail

# Same-source-checkpoint baseline suite on ACDC.
#
# All methods run on the SAME Panoptic-FPN MTL checkpoint that CTCMT_MTL uses,
# so CTTA algorithms are compared under identical source weights. Checkpoint
# compatibility was verified: PFN contains all 307 GeneralizedRCNN keys and all
# 304 SemanticSegmentor keys with identical shapes.
#
#   bash scripts/run_same_source_baselines_acdc.sh [GPU] [SEED]

GPU="${1:-1}"
SEED="${2:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFGDIR="detectron2/configs/Cityscapes"

DET_STREAM='("acdc_fog","acdc_night","acdc_rain","acdc_snow")'
SEG_STREAM='("acdc_fog_semseg","acdc_night_semseg","acdc_rain_semseg","acdc_snow_semseg")'

echo ">>>>>> AMROD on PFN source (detection)"
STREAM="${DET_STREAM}" bash "${HERE}/run_ctcr_acdc_local.sh" \
  "${GPU}" "amrod_pfnsrc_seed${SEED}" "${CFGDIR}/amrod_pfn_R_50_ACDC.yaml" "${SEED}"

echo ">>>>>> CoTTA on PFN source (segmentation)"
STREAM="${SEG_STREAM}" bash "${HERE}/run_ctcr_acdc_local.sh" \
  "${GPU}" "cotta_pfnsrc_seed${SEED}" "${CFGDIR}/cotta_semseg_pfn_R_50_ACDC.yaml" "${SEED}"

echo ">>>>>> TENT on PFN source (segmentation)"
STREAM="${SEG_STREAM}" bash "${HERE}/run_ctcr_acdc_local.sh" \
  "${GPU}" "tent_pfnsrc_seed${SEED}" "${CFGDIR}/tent_semseg_pfn_R_50_ACDC.yaml" "${SEED}"

echo "SAME-SOURCE BASELINE SUITE DONE (seed ${SEED})"
