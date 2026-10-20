#!/usr/bin/env bash
# Smoke test for Table 4 ablation seed runs (no-CL variants).
# Same pattern as smoke_test.sh: fog-only, timeout 240, tail 80.
set -euo pipefail
GPU=${1:-0}

CONFIGS=(
  "detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_seed0_pfn_R_50_ACDC.yaml"
  "detectron2/configs/Cityscapes/ctcmt_mtl_no_ctcl_v2_seed0_pfn_R_50_ACDC.yaml"
)

for CFG in "${CONFIGS[@]}"; do
  NAME="$(basename "${CFG}" .yaml)"
  OUT="/workspace/output/smoke_${NAME}"
  LOG="/tmp/smoke_${NAME}.log"
  echo "=== SMOKE: ${NAME} (streaming; also tee -> ${LOG}) ==="
  docker run --rm --gpus "\"device=${GPU}\"" --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -v /home/ilias/AMROD:/workspace/amrod \
    -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
    -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
    -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
    -v /data/ilias/acdc:/datasets/ACDC:ro \
    -v /data/vgcmt/datasets/cityscapes_foggy:/datasets/cityscapes_foggy:ro \
    -v /data/ilias/panoptic_fpn/output:/workspace/output \
    -w /workspace/amrod \
    -e DETECTRON2_DATASETS=/datasets \
    -e PYTHONPATH=/workspace/amrod/detectron2 \
    amrod:latest bash -c "
      export HOME=/tmp
      pip install --quiet --user --no-warn-script-location shapely
      timeout 240 python -u detectron2/tools/train_net.py \
        --config-file ${CFG} \
        --eval-only --num-gpus 1 \
        DATASETS.TEST '(\"acdc_fog_mtl\",)' \
        OUTPUT_DIR ${OUT}
    " 2>&1 | tee "${LOG}" || { echo "SMOKE_FAIL: ${NAME}"; exit 1; }
  echo "=== SMOKE OK: ${NAME} ==="
done

echo "ALL SMOKE TESTS PASSED"
