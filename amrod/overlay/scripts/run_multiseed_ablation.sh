#!/usr/bin/env bash
# Run multi-seed ablation for the CT-CL vs CT-CL+V2 variance study.
# 3 seeds × 2 configs = 6 runs. Confirms the 0.32 AP50 gap between early
# CTCMT_MTL_v4 (36.03) and overnight CT-CL-no-V2 (36.35) is within noise.
#
# Usage: bash scripts/run_multiseed_ablation.sh [START_GPU]
# Runs three pairs of jobs (seeds 0/42/123) sequentially on START_GPU.
set -euo pipefail

GPU=${1:-1}
OUT_ROOT=/workspace/output/ctta_acdc/multiseed

DOCKER_RUN="docker run --rm --gpus \"device=${GPU}\" --shm-size=8g
  --user $(id -u):$(id -g)
  -v /home/ilias/AMROD:/workspace/amrod
  -v /data/vgcmt/datasets/cityscapes:/data/vgcmt/datasets/cityscapes:ro
  -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro
  -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro
  -v /data/ilias/acdc:/datasets/ACDC:ro
  -v /data/ilias/panoptic_fpn/output:/workspace/output
  -w /workspace/amrod
  -e DETECTRON2_DATASETS=/datasets
  -e PYTHONPATH=/workspace/amrod/detectron2
  amrod:latest"

_run() {
  local tag="$1" cfg="$2" outdir="$3"
  echo ">>> Starting ${tag} ..."
  eval docker run --rm \
    --gpus "\"device=${GPU}\"" \
    --shm-size=8g \
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
    amrod:latest bash -c \'"
      export HOME=/tmp
      pip install --quiet --user shapely 2>&1 | tail -1
      python detectron2/tools/train_net.py \
        --config-file ${cfg} \
        --eval-only --num-gpus 1 \
        OUTPUT_DIR ${outdir} 2>&1
      echo DONE_${tag}
    "\'
  echo "<<< ${tag} finished"
}

for seed in 0 42 123; do
  _run "ctcl_no_v2_seed${seed}" \
    "detectron2/configs/Cityscapes/ctcmt_mtl_ctcl_no_v2_seed${seed}_pfn_R_50_ACDC.yaml" \
    "${OUT_ROOT}/ctcl_no_v2_seed${seed}"

  _run "ctcmt_v2_seed${seed}" \
    "detectron2/configs/Cityscapes/ctcmt_v2_seed${seed}_pfn_R_50_ACDC.yaml" \
    "${OUT_ROOT}/ctcmt_v2_seed${seed}"
done

echo "All 6 multi-seed runs complete. Results in /data/ilias/panoptic_fpn/output/ctta_acdc/multiseed/"
