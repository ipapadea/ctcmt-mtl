#!/usr/bin/env bash
set -euo pipefail

# CT-CR 2x2 completion: run variant D (soft_seg_global) on ACDC, 3 seeds,
# sequentially on one GPU.
#
#   bash scripts/run_d_acdc_3seeds.sh [GPU]

GPU="${1:-1}"
CFG="detectron2/configs/Cityscapes/ctcmt_e8d_v2_softglobal_ctcr_no_ctpv.yaml"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for SEED in 0 42 123; do
  echo ">>>>>> D / ACDC / seed ${SEED} / GPU ${GPU}"
  bash "${HERE}/run_ctcr_acdc_local.sh" "${GPU}" "ctcr_D_acdc_seed${SEED}" "${CFG}" "${SEED}"
done

echo "ALL D ACDC SEEDS DONE"
