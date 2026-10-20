#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

export HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
export HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"

METHODS=(
  "lambda0|detectron2/configs/Cityscapes/newsrc_csc_ctcr_lambda0_probe.yaml"
  "lambda1|detectron2/configs/Cityscapes/newsrc_csc_ctcr.yaml"
  "taskcoco|detectron2/configs/Cityscapes/newsrc_csc_taskcoco.yaml"
  "base|detectron2/configs/Cityscapes/newsrc_csc_base.yaml"
)

CHALLENGERS=(
  motion_blur_mtl
  snow_mtl
  brightness_mtl
  defocus_blur_mtl
)

for SPEC in "${METHODS[@]}"; do
    METHOD="${SPEC%%|*}"
    CFG="${SPEC#*|}"

    for X in "${CHALLENGERS[@]}"; do
        SHORT="${X%_mtl}"

        EXP="retention_pair_fog_${SHORT}_${METHOD}_s0"

        echo
        echo "################################################################"
        echo "${EXP}"
        echo "################################################################"

        bash scripts/run_retention_probe_local.sh \
          "${GPU}" \
          "${EXP}" \
          "${CFG}" \
          "${METHOD}" \
          0 \
          --rounds 1 \
          --cycle "fog_mtl,${X}" \
          --probe-after-each "fog_mtl"
    done
done
