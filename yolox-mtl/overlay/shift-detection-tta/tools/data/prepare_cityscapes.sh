#!/usr/bin/env bash
# Prepare a Cityscapes dataset directory for the MTL YOLOX config.
#
# Expected input: two zip files from https://www.cityscapes-dataset.com/downloads/
#   - leftImg8bit_trainvaltest.zip  (RGB images, ~11 GB)
#   - gtFine_trainvaltest.zip       (fine annotations, ~241 MB)
#
# Output layout (matches configs/source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py):
#
#   ${OUT_DIR}/
#   ├── leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
#   ├── gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png            (from zip)
#   ├── gtFine/{train,val,test}/<city>/*_gtFine_labelTrainIds.png       (generated)
#   ├── gtFine/{train,val,test}/<city>/*_gtFine_instanceIds.png         (from zip)
#   ├── gtFine/{train,val,test}/<city>/*_gtFine_polygons.json           (from zip)
#   ├── instancesonly_filtered_gtFine_train.json                        (generated)
#   ├── instancesonly_filtered_gtFine_val.json                          (generated)
#   └── instancesonly_filtered_gtFine_test.json                         (generated)
#
# Usage (from repo root, INSIDE the container via `docker/run.sh`):
#
#   tools/data/prepare_cityscapes.sh \
#       --zips-dir /workspace/data/downloads/cityscapes \
#       --out-dir  /workspace/data/cityscapes \
#       --nproc 8
#
# Or set env vars: ZIPS_DIR, OUT_DIR, NPROC. Env vars are only read if the
# equivalent flag isn't passed.
#
# Steps are idempotent: rerunning after a partial prep only redoes missing parts.

set -euo pipefail

ZIPS_DIR="${ZIPS_DIR:-}"
OUT_DIR="${OUT_DIR:-/workspace/data/cityscapes}"
NPROC="${NPROC:-4}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zips-dir) ZIPS_DIR="$2"; shift 2;;
        --out-dir)  OUT_DIR="$2";  shift 2;;
        --nproc)    NPROC="$2";    shift 2;;
        -h|--help)
            sed -n '2,32p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$ZIPS_DIR" ]]; then
    echo "error: --zips-dir (or ZIPS_DIR env) is required" >&2
    exit 2
fi

LEFT_ZIP="$ZIPS_DIR/leftImg8bit_trainvaltest.zip"
GT_ZIP="$ZIPS_DIR/gtFine_trainvaltest.zip"
for z in "$LEFT_ZIP" "$GT_ZIP"; do
    if [[ ! -f "$z" ]]; then
        echo "error: missing zip: $z" >&2
        echo "hint : sign up at cityscapes-dataset.com and download the two"
        echo "       trainvaltest zips into $ZIPS_DIR/" >&2
        exit 3
    fi
done

mkdir -p "$OUT_DIR"

# ------- Step 1: unzip (skip if already extracted) --------------------------
# `-o` = overwrite without prompting (avoids interactive prompts on partial retries)
if [[ ! -d "$OUT_DIR/leftImg8bit/train" ]]; then
    echo "[1/3] unzipping leftImg8bit_trainvaltest.zip -> $OUT_DIR/"
    unzip -oq "$LEFT_ZIP" -d "$OUT_DIR"
else
    echo "[1/3] leftImg8bit/train already present, skipping unzip"
fi
if [[ ! -d "$OUT_DIR/gtFine/train" ]]; then
    echo "     unzipping gtFine_trainvaltest.zip"
    unzip -oq "$GT_ZIP" -d "$OUT_DIR"
else
    echo "     gtFine/train already present, skipping unzip"
fi

# ------- Step 2: generate *_labelTrainIds.png -------------------------------
# createTrainIdLabelImgs.py walks CITYSCAPES_DATASET and writes a sibling
# *_labelTrainIds.png next to each *_labelIds.png.
FIRST_TRAIN_ID_MASK="$(find "$OUT_DIR/gtFine/train" -maxdepth 3 -name '*_gtFine_labelTrainIds.png' | head -1 || true)"
if [[ -z "$FIRST_TRAIN_ID_MASK" ]]; then
    echo "[2/3] generating *_gtFine_labelTrainIds.png masks"
    CITYSCAPES_DATASET="$OUT_DIR" \
        python -m cityscapesscripts.preparation.createTrainIdLabelImgs
else
    echo "[2/3] labelTrainIds masks already present, skipping generation"
fi

# ------- Step 3: bbox COCO JSONs via mmdet's converter ----------------------
NEED_CONVERT=0
for split in train val; do
    if [[ ! -f "$OUT_DIR/instancesonly_filtered_gtFine_${split}.json" ]]; then
        NEED_CONVERT=1
    fi
done
if [[ "$NEED_CONVERT" -eq 1 ]]; then
    echo "[3/3] converting Cityscapes instances -> COCO bbox json (nproc=$NPROC)"
    python /opt/conda/lib/python3.9/site-packages/mmdet/.mim/tools/dataset_converters/cityscapes.py \
        "$OUT_DIR" --nproc "$NPROC" -o "$OUT_DIR"
else
    echo "[3/3] COCO bbox jsons already present, skipping conversion"
fi

echo
echo "Cityscapes prep complete."
echo "  root : $OUT_DIR"
echo "Now train the MTL source model:"
echo "  tools/train.py configs/source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py"
