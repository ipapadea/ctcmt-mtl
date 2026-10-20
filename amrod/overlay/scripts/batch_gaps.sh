#!/usr/bin/env bash
set -euo pipefail

# Closes the two remaining gaps in the results tables, on a single GPU, ~4.25 h.
#
#   1. ACDC 4-domain (AMROD Table 4 protocol) with the current method, 3 seeds.
#      The 38.75 currently in that table is a one-off e10 run and the n=3 row
#      (37.07) is the old base, so neither matches the method we report.
#   2. Cityscapes-C 12-corruption seeds 42 and 123. Seed 0 gave 17.011 AP50
#      (a win over AMROD's 14.984) but n=1 is not reportable.
#
# Cheap ACDC runs go first so that table row exists within ~75 min even if the
# batch is interrupted.
#
#   bash scripts/batch_gaps.sh [GPU] [CONFIG]

GPU="${1:-0}"
CFG="${2:-detectron2/configs/Cityscapes/ctcmt_e11_bothsc_ctcrD.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag=$(basename "${CFG}" .yaml | sed 's/^ctcmt_//')

echo "######################################################################"
echo "GAP BATCH   config=${tag}   gpu=${GPU}"
echo "  1-3) ACDC 4-domain   seeds 0/42/123   ~25 min each"
echo "  4-5) CS-C 12-corr    seeds 42/123     ~90 min each"
echo "######################################################################"

for SEED in 0 42 123; do
  echo "###### ACDC 4-domain  ${tag}  seed ${SEED}"
  bash "${HERE}/run_ctcr_acdc_local.sh" "${GPU}" "${tag}_acdc4_s${SEED}" "${CFG}" "${SEED}" \
    || echo "!!! acdc4 seed ${SEED} failed, continuing"
done

for SEED in 42 123; do
  echo "###### CS-C 12-corr  ${tag}  seed ${SEED}"
  bash "${HERE}/run_csc12_local.sh" "${GPU}" "${tag}_csc12_s${SEED}" "${CFG}" "${SEED}" \
    || echo "!!! csc12 seed ${SEED} failed, continuing"
done

echo
echo "GAP BATCH DONE. Refresh the tables with:"
echo "  python3 scripts/gather_results.py --logs /media/ilias/DATA/ilias/amrod_output/logs"
