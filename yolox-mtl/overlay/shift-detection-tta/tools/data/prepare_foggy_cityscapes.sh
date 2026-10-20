#!/usr/bin/env bash
# Prepare Foggy Cityscapes (Sakaridis et al. 2018) for the CT-CMT
# Cityscapes -> Foggy Cityscapes CTTA config.
#
# Requires that clear Cityscapes is already prepped (see prepare_cityscapes.sh):
#   ${CS_OUT_DIR}/leftImg8bit/val/<city>/*.png
#   ${CS_OUT_DIR}/gtFine/val/<city>/*_gtFine_labelTrainIds.png
#   ${CS_OUT_DIR}/instancesonly_filtered_gtFine_val.json
#
# Expected input zip (from https://www.cityscapes-dataset.com/downloads/):
#   leftImg8bit_trainvaltest_foggyDBF.zip   (~30 GB, all three betas)
# (or the val-only variant, if that's what you downloaded).
#
# Output layout (matches ctcmt_yolox_mtl_cityscapes_to_foggy.py):
#
#   ${OUT_DIR}/
#   ├── leftImg8bit_foggyDBF/val/<city>/*_leftImg8bit_foggy_beta_{0.005,0.01,0.02}.png
#   ├── instancesonly_filtered_gtFine_val_foggy_beta_0.005.json
#   ├── instancesonly_filtered_gtFine_val_foggy_beta_0.01.json
#   └── instancesonly_filtered_gtFine_val_foggy_beta_0.02.json
#
# Usage (INSIDE the container via docker/run.sh):
#
#   tools/data/prepare_foggy_cityscapes.sh \
#       --zip     /workspace/data/downloads/leftImg8bit_trainvaltest_foggyDBF.zip \
#       --cs-dir  /workspace/data/cityscapes \
#       --out-dir /workspace/data/cityscapes_foggy
#
# Env-var equivalents: ZIP, CS_DIR, OUT_DIR. Idempotent.

set -euo pipefail

ZIP="${ZIP:-}"
CS_DIR="${CS_DIR:-/workspace/data/cityscapes}"
OUT_DIR="${OUT_DIR:-/workspace/data/cityscapes_foggy}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip)     ZIP="$2";     shift 2;;
        --cs-dir)  CS_DIR="$2";  shift 2;;
        --out-dir) OUT_DIR="$2"; shift 2;;
        -h|--help) sed -n '2,32p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ ! -f "$CS_DIR/instancesonly_filtered_gtFine_val.json" ]]; then
    echo "error: clear Cityscapes not prepped at $CS_DIR" >&2
    echo "       run tools/data/prepare_cityscapes.sh first" >&2
    exit 3
fi

mkdir -p "$OUT_DIR"

# ------- Step 1: unzip Foggy zip (skip if already extracted) ---------------
if [[ ! -d "$OUT_DIR/leftImg8bit_foggyDBF/val" ]]; then
    if [[ -z "$ZIP" || ! -f "$ZIP" ]]; then
        echo "error: --zip must point at an existing zip file" >&2
        echo "hint : need leftImg8bit_trainvaltest_foggyDBF.zip from" >&2
        echo "       https://www.cityscapes-dataset.com/downloads/" >&2
        exit 3
    fi
    echo "[1/2] unzipping $ZIP -> $OUT_DIR/"
    unzip -q "$ZIP" -d "$OUT_DIR"
else
    echo "[1/2] leftImg8bit_foggyDBF/val already present, skipping unzip"
fi

# ------- Step 2: per-beta COCO jsons (rewrite file_names) ------------------
NEED_JSON=0
for beta in 0.005 0.01 0.02; do
    if [[ ! -f "$OUT_DIR/instancesonly_filtered_gtFine_val_foggy_beta_${beta}.json" ]]; then
        NEED_JSON=1
    fi
done
if [[ "$NEED_JSON" -eq 1 ]]; then
    echo "[2/2] emitting per-beta COCO jsons"
    python tools/data/make_foggy_cityscapes_json.py \
        --clear-json "$CS_DIR/instancesonly_filtered_gtFine_val.json" \
        --out-dir    "$OUT_DIR"
else
    echo "[2/2] per-beta jsons already present, skipping"
fi

# Sanity check: one image + json entry per beta.
for beta in 0.005 0.01 0.02; do
    n_img=$(find "$OUT_DIR/leftImg8bit_foggyDBF/val" -name "*_beta_${beta}.png" | wc -l)
    n_json=$(python -c "
import json,sys
d=json.load(open('$OUT_DIR/instancesonly_filtered_gtFine_val_foggy_beta_${beta}.json'))
print(len(d['images']))
")
    echo "  beta=$beta  images_on_disk=$n_img  images_in_json=$n_json"
done

echo
echo "Foggy Cityscapes prep complete."
echo "  root : $OUT_DIR"
echo "Now run continual TTA:"
echo "  tools/test.py configs/continuous/ctcmt/ctcmt_yolox_mtl_cityscapes_to_foggy.py \\"
echo "      --checkpoint work_dirs/yolox_x_mtl_8xb4-24e_cityscapes/epoch_24.pth"
