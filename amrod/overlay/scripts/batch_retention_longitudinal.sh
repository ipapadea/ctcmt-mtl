#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

export HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
export HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"

run_one () {
    METHOD="$1"
    CFG="$2"

    EXP="retention_long_${METHOD}_r3_s0"

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
      --rounds 3 \
      --probe-after-each "fog_mtl" \
      --round-end-all
}

run_one \
  lambda0 \
  detectron2/configs/Cityscapes/newsrc_csc_ctcr_lambda0_probe.yaml

run_one \
  lambda1 \
  detectron2/configs/Cityscapes/newsrc_csc_ctcr.yaml

run_one \
  taskcoco \
  detectron2/configs/Cityscapes/newsrc_csc_taskcoco.yaml

run_one \
  base \
  detectron2/configs/Cityscapes/newsrc_csc_base.yaml
