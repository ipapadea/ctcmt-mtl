#!/usr/bin/env bash
set -euo pipefail

# Cityscapes-C long-term: the last open question, plus an independent check of
# the number it is measured against.
#
#   1. E14 = e11 with the segmentation branch switched off (one factor).
#      Tests shared-trunk negative transfer -- the only structural difference
#      left between us and AMROD. Five other explanations are already falsified
#      (gate closure, score-EMA drift, label starvation, threshold saturation,
#      AMROD's restoration rule).
#
#   2. AMROD on the PFN checkpoint, re-run HERE. Its 26.256 came from one seed
#      on the remote machine; this reproduces it on Cronus, giving both a
#      replication check and a same-hardware comparison.
#
# ~11 h total on one GPU. Reference: e11 = 23.69 +/- 0.67, plateau ~26 from R5.
#
#   bash scripts/batch_negative_transfer.sh [GPU]

GPU="${1:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes

CYCLE=(fog motion_blur snow brightness defocus_blur)
mtl_stream="("; det_stream="("
for _ in $(seq 1 10); do
  for c in "${CYCLE[@]}"; do
    mtl_stream+="\"${c}_mtl\","
    det_stream+="\"${c}\","          # AMROD is detection-only: no sem_seg evaluator
  done
done
mtl_stream="${mtl_stream%,})"; det_stream="${det_stream%,})"

echo "######################################################################"
echo "NEGATIVE-TRANSFER BATCH   gpu=${GPU}"
echo "  1) e14  det-only (seg branch off)   ~5.5 h"
echo "  2) AMROD on PFN, replication        ~5.5 h"
echo "reference: e11 23.69 +/- 0.67 | AMROD (remote) 26.256"
echo "######################################################################"

echo "###### 1/2  e14 det-only  CS-C x10  seed 0"
STREAM="${mtl_stream}" bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU}" "e14_detonly_cscLT_s0" "${CFG}/ctcmt_e14_detonly.yaml" 0 \
  || echo "!!! e14 failed, continuing"

echo "###### 2/2  AMROD on PFN  CS-C x10  seed 0  (replication)"
STREAM="${det_stream}" bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU}" "amrod_pfnsrc_cscLT_cronus_s0" "${CFG}/amrod_pfn_R_50_CS_C.yaml" 0 \
  || echo "!!! amrod replication failed, continuing"

echo
echo "BATCH DONE. Compare with:"
echo "  python3 scripts/gather_results.py --logs /media/ilias/DATA/ilias/amrod_output/logs"
echo "  python3 scripts/diagnose_plateau.py /media/ilias/DATA/ilias/amrod_output/logs/e14_detonly_cscLT_s0.log"
