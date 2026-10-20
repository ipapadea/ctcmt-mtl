#!/usr/bin/env bash
# Overnight batch: all remaining ablations + Foggy Cityscapes benchmark.
# GPU 0: Table 2 ablation (CTCMT_Seg component isolation)
# GPU 1: Table 4 ablation (CTCMT_MTL V2 contribution isolation)
# GPU 2: Foggy Cityscapes benchmark (source-only + AMROD + AMROD+V2)
# GPU 3: Table 2 ablation continued + sanity

LOGS=/data/ilias/panoptic_fpn/output/ctta_acdc/logs_overnight
mkdir -p "$LOGS"

tmux kill-server 2>/dev/null || true
RUNNER="bash /home/ilias/AMROD/scripts/run_ctta_acdc.sh"

# GPU 0: Table 2 ablation — two missing rows
tmux new-session -d -s night_seg_abl \
  "echo '[GPU0] Table 2 ablation' && \
   $RUNNER 0 ctcmt_seg_proto_only > $LOGS/seg_proto_only.log 2>&1 && \
   $RUNNER 0 ctcmt_seg_small_aug_no_v2 > $LOGS/seg_small_aug_no_v2.log 2>&1 && \
   echo GPU0_DONE"

# GPU 1: Table 4 ablation — V2 contribution to MTL
tmux new-session -d -s night_mtl_abl \
  "echo '[GPU1] Table 4 ablation' && \
   $RUNNER 1 ctcmt_mtl_no_ctcl_v2 > $LOGS/mtl_no_ctcl_v2.log 2>&1 && \
   $RUNNER 1 ctcmt_mtl_ctcl_no_v2 > $LOGS/mtl_ctcl_no_v2.log 2>&1 && \
   echo GPU1_DONE"

# GPU 2: Foggy Cityscapes (source-only + AMROD + AMROD+V2)
tmux new-session -d -s night_foggy \
  "echo '[GPU2] Foggy Cityscapes' && \
   $RUNNER 2 source_only_foggy > $LOGS/foggy_source_only.log 2>&1 && \
   $RUNNER 2 amrod_foggy > $LOGS/foggy_amrod.log 2>&1 && \
   $RUNNER 2 amrod_v2_foggy > $LOGS/foggy_amrod_v2.log 2>&1 && \
   echo GPU2_DONE"

# GPU 3: CTCMT_MTL+V2 on Foggy Cityscapes + CTCMT_MTL source-only
tmux new-session -d -s night_foggy2 \
  "echo '[GPU3] Foggy MTL' && \
   docker run --rm --gpus '\"device=3\"' --shm-size=8g \
     --user \"\$(id -u):\$(id -g)\" \
     -v /home/ilias/AMROD:/workspace/amrod \
     -v /data/vgcmt/datasets:/datasets \
     -v /data/ilias/cityscapes_pfn:/datasets/cityscapes:ro \
     -v /data/ilias/panoptic_fpn/output/coco_annotations:/datasets/annotations:ro \
     -v /data/ilias/panoptic_fpn/output:/workspace/output \
     -w /workspace/amrod \
     -e DETECTRON2_DATASETS=/datasets \
     -e PYTHONPATH=/workspace/amrod/detectron2 \
     amrod:latest bash -c '
       export HOME=/tmp
       pip install --quiet --user --no-warn-script-location shapely 2>&1 | tail -1
       echo \">>> source-only PFN on foggy_cityscapes_val <<<\"
       python detectron2/tools/train_net.py \
         --config-file detectron2/configs/Cityscapes/panoptic_fpn_R_50.yaml \
         --eval-only --num-gpus 1 \
         MODEL.WEIGHTS /workspace/output/panoptic_fpn_R50_cityscapes/model_final.pth \
         DATASETS.TEST \"(\\\"foggy_cityscapes_val\\\",)\" \
         OUTPUT_DIR /workspace/output/source_only_pfn_foggy
       echo \">>> CTCMT_MTL+V2 on foggy_cityscapes_val <<<\"
       python detectron2/tools/train_net.py \
         --config-file detectron2/configs/Cityscapes/ctcmt_v2_cross_fisher_pfn_R_50_ACDC.yaml \
         --eval-only --num-gpus 1 \
         DATASETS.TEST \"(\\\"foggy_cityscapes_val\\\",)\" \
         OUTPUT_DIR /workspace/output/ctcmt_v2_foggy
     ' > $LOGS/foggy_pfn_and_ctcmt.log 2>&1 && \
   echo GPU3_DONE"

sleep 30
tmux ls
echo "All 4 GPUs launched. Logs at $LOGS"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
