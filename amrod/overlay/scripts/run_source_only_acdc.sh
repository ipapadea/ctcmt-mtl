#!/usr/bin/env bash
set -euo pipefail
GPU="$1"
TRACK="$2"        # maskrcnn | semseg | pfn
CFG=""
CKPT=""
DSET_SUFFIX=""
case "$TRACK" in
  maskrcnn)
    CFG="detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN_singletask.yaml"
    CKPT="/workspace/output/mask_rcnn_R50_cityscapes/model_final.pth"
    DSET_SUFFIX=""
    ;;
  semseg)
    CFG="detectron2/configs/Cityscapes/semantic_R_50_FPN_singletask.yaml"
    CKPT="/workspace/output/semantic_R50_cityscapes/model_final.pth"
    DSET_SUFFIX="_semseg"
    ;;
  pfn)
    CFG="detectron2/configs/Cityscapes/panoptic_fpn_R_50.yaml"
    CKPT="/workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth"
    DSET_SUFFIX="_mtl"
    ;;
  *) echo "unknown track $TRACK"; exit 1 ;;
esac

docker run --rm --gpus "\"device=${GPU}\"" --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -v /home/ilias/AMROD:/workspace/amrod \
  -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro \
  -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
  -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
  -v /data/ilias/acdc:/datasets/ACDC:ro \
  -v /data/ilias/panoptic_fpn/output:/workspace/output \
  -w /workspace/amrod \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  amrod:latest bash -c "
    export HOME=/tmp
    pip install --quiet --user --no-warn-script-location shapely 2>&1 | tail -1
    for W in fog night rain snow; do
      echo '>>> ${TRACK} source-only on acdc_'\${W}\"${DSET_SUFFIX}\"' <<<'
      python detectron2/tools/train_net.py \
        --config-file ${CFG} \
        --eval-only --num-gpus 1 \
        MODEL.WEIGHTS ${CKPT} \
        DATASETS.TEST '(\"acdc_'\${W}'${DSET_SUFFIX}\",)' \
        OUTPUT_DIR /workspace/output/source_only_acdc/${TRACK}/\${W} 2>&1 | grep -E 'copypaste:|Task: |Loading|IoU,|:iIoU'
      echo ''
    done
    echo SRC_ONLY_${TRACK}_DONE
  "
