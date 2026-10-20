#!/usr/bin/env bash
set -euo pipefail

# Overnight batch, GPU 0: re-establish the headline ACDC long-term table on the
# CURRENT method configuration (e11). The published 41.30 +/- 0.79 was produced
# by the old base (no strong augmentation, no class balancing, CT-CR A), so it
# no longer corresponds to the method we describe.
#
# 3 seeds x ~3.5 h = ~10.5 h
#
#   bash scripts/batch_night_gpu0.sh [GPU] [CONFIG]

GPU="${1:-0}"
CFG="${2:-detectron2/configs/Cityscapes/ctcmt_e11_bothsc_ctcrD.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CYCLE=(fog night rain snow)
stream="("
for _ in $(seq 1 10); do for c in "${CYCLE[@]}"; do stream+="\"acdc_${c}_mtl\","; done; done
stream="${stream%,})"

tag=$(basename "${CFG}" .yaml | sed 's/^ctcmt_//')
for SEED in 0 42 123; do
  echo "###### ACDC x10  ${tag}  seed ${SEED}  (GPU ${GPU})"
  STREAM="${stream}" bash "${HERE}/run_ctcr_acdc_local.sh" \
    "${GPU}" "${tag}_acdcLT_s${SEED}" "${CFG}" "${SEED}" || echo "!!! seed ${SEED} failed, continuing"
done

echo "BATCH GPU${GPU} DONE — ACDC x10 seeds 0/42/123 for ${tag}"
