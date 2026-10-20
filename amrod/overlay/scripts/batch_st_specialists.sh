#!/usr/bin/env bash
set -euo pipefail

# Specialist / source-model study on the Cityscapes-C long-term stream:
#   (fog -> motion_blur -> snow -> brightness -> defocus_blur) x 10, no reset.
#
#   ST-D  Mask R-CNN R50-FPN specialist + our detection CTTA   (cf. E15)
#   ST-S  Semantic FPN R50   specialist + our semantic  CTTA   (cf. E21)
#
# These CHANGE THE SOURCE CHECKPOINT. They are not same-source with
# E13a/E15/E21 and must be reported as a separate study, never inside the main
# ablation table. Same-source references, seed 0, for reading the result:
#
#            AP50    mIoU
#   e13a     24.80   30.83   full MTL
#   e15      26.80   35.51   det-only   <- ST-D's counterpart
#   e21      15.10   28.21   seg-only   <- ST-S's counterpart
#   source   13.17   27.21   no adaptation
#
# ST-S is the one that matters: E21's mIoU peaks at round 4 (29.2) then decays
# to 26.7. If the Semantic FPN specialist decays the same way, the segmentation
# self-distillation is intrinsically unstable and architecture-independent.
#
#   bash scripts/batch_st_specialists.sh [GPU_A] [GPU_B]
#
# Defaults below are the gpu1 paths; override per machine.

GPU_A="${1:-0}"
GPU_B="${2:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=detectron2/configs/Cityscapes
HOST_OUT="${HOST_OUT:-/data/ilias/amrod_output}"

# Preflight: WEIGHTS are resolved inside the container under HOST_OUT. A wrong
# or unset HOST_OUT makes docker create an empty dir and every arm dies on a
# missing checkpoint minutes later. Copy them from Cronus if absent:
#   rsync -avP --relative \
#     /media/ilias/DATA/ilias/amrod_output/./mask_rcnn_R50_cityscapes/model_final.pth \
#     /media/ilias/DATA/ilias/amrod_output/./semantic_R50_cityscapes/model_final.pth \
#     ilias@gpu1:/data/ilias/amrod_output/
for ckpt in mask_rcnn_R50_cityscapes semantic_R50_cityscapes; do
  [[ -f "${HOST_OUT}/${ckpt}/model_final.pth" ]] || {
    echo "ERROR: missing ${HOST_OUT}/${ckpt}/model_final.pth (HOST_OUT=${HOST_OUT})" >&2
    exit 2; }
done

CYCLE=(fog motion_blur snow brightness defocus_blur)
det_stream="("; seg_stream="("
for _ in $(seq 1 10); do
  for c in "${CYCLE[@]}"; do
    det_stream+="\"${c}\","          # Mask R-CNN emits no sem_seg
    seg_stream+="\"${c}_semseg\","   # Semantic FPN emits no detections
  done
done
det_stream="${det_stream%,})"; seg_stream="${seg_stream%,})"

echo "######################################################################"
echo "SPECIALIST STUDY   seed 0   CS-C x10 (50 evals each, no reset)"
echo "  gpu ${GPU_A}: ST-D std_mrcnn_cscLT_s0"
echo "  gpu ${GPU_B}: ST-S sts_semfpn_cscLT_s0"
echo "  refs: e15 26.80 AP50 | e21 28.21 mIoU (peak 29.2 R4 -> 26.7 R10)"
echo "######################################################################"

STREAM="${det_stream}" bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU_A}" std_mrcnn_cscLT_s0 "${CFG}/ctcmt_std_mrcnn_cscLT.yaml" 0 & pid_a=$!
STREAM="${seg_stream}" bash "${HERE}/run_mixed_lt_local.sh" \
  "${GPU_B}" sts_semfpn_cscLT_s0 "${CFG}/ctcmt_sts_semfpn_cscLT.yaml" 0 & pid_b=$!
wait "${pid_a}" || echo "!!! ST-D failed"
wait "${pid_b}" || echo "!!! ST-S failed"

echo
echo "DONE. Both must show 50 evals and 0 tracebacks:"
echo "  for f in std_mrcnn_cscLT_s0 sts_semfpn_cscLT_s0; do"
echo "    echo \"\$f: \$(grep -c 'in csv format' ${HOST_OUT}/logs/\$f.log) evals,"
echo "      \$(grep -c Traceback ${HOST_OUT}/logs/\$f.log) tracebacks\"; done"
