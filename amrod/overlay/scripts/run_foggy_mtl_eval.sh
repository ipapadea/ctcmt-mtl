#!/usr/bin/env bash
# Re-evaluate CTCMT_MTL+V2 on Foggy Cityscapes with BOTH bbox AP and sem mIoU.
# Uses the new foggy_cityscapes_val_mtl dataset (det + Cityscapes GT sem labels).
set -euo pipefail

GPU=${1:-2}

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
    pip install --quiet --user shapely 2>&1 | tail -1
    echo '>>> CTCMT_MTL+V2 Foggy Cityscapes — bbox + seg mIoU <<<'
    python detectron2/tools/train_net.py \
      --config-file detectron2/configs/Cityscapes/ctcmt_v2_foggy_cityscapes_mtl.yaml \
      --eval-only --num-gpus 1 \
      2>&1
    echo FOGGY_MTL_DONE
  "
