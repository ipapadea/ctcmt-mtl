#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
CFGDIR="detectron2/configs/Cityscapes"

CYCLE=(fog night rain snow)

STREAM="("
for _ in $(seq 1 10); do
    for c in "${CYCLE[@]}"; do
        STREAM+="\"acdc_${c}_mtl\","
    done
done
STREAM="${STREAM%,})"

export STREAM

for ARM in base ctcl ctcr ctcl_ctcr ctcl_ctcr_ctpv; do
    echo
    echo "============================================================"
    echo "ACDC LT -- ${ARM}"
    echo "============================================================"

    bash scripts/run_ctcr_acdc_local.sh \
        "${GPU}" \
        "newsrc_${ARM}_acdcLT_s0" \
        "${CFGDIR}/newsrc_acdc_${ARM}.yaml" \
        0
done
