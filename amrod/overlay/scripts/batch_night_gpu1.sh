#!/usr/bin/env bash
set -euo pipefail

# Overnight batch, GPU 1: finish Cityscapes-C on the current method (e11).
#   - CS-C long-term seeds 42 and 123  (seed 0 already done: 24.580 AP50)
#   - CS-C short-term, 12 corruptions  (AMROD Table 2 — no run exists on the
#     new base at all; the 14.67 on record is from the old base)
#
# ~5.5 + 1.5 + 5.5 = ~12.5 h. The short run is placed second so the missing
# table row exists early even if the night is cut short.
#
#   bash scripts/batch_night_gpu1.sh [GPU] [CONFIG]

GPU="${1:-1}"
CFG="${2:-detectron2/configs/Cityscapes/ctcmt_e11_bothsc_ctcrD.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag=$(basename "${CFG}" .yaml | sed 's/^ctcmt_//')

echo "###### CS-C x10  ${tag}  seed 42  (GPU ${GPU})"
bash "${HERE}/run_mixed_lt_local.sh" "${GPU}" "${tag}_cscLT_s42" "${CFG}" 42 \
  || echo "!!! cscLT seed42 failed, continuing"

echo "###### CS-C 12-corr  ${tag}  seed 0  (GPU ${GPU})"
bash "${HERE}/run_csc12_local.sh" "${GPU}" "${tag}_csc12_s0" "${CFG}" 0 \
  || echo "!!! csc12 seed0 failed, continuing"

echo "###### CS-C x10  ${tag}  seed 123  (GPU ${GPU})"
bash "${HERE}/run_mixed_lt_local.sh" "${GPU}" "${tag}_cscLT_s123" "${CFG}" 123 \
  || echo "!!! cscLT seed123 failed, continuing"

echo "BATCH GPU${GPU} DONE — CS-C LT s42/s123 + CS-C 12corr s0 for ${tag}"
