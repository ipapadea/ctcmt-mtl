#!/usr/bin/env bash
set -Eeuo pipefail

# One-way, resumable copy of everything the MTCLSR batch needs onto the remote machine
# (no docker there, see scripts/remote_env_leviathan.sh).  Safe to re-run: rsync only
# sends what is missing/changed and never deletes anything on the remote.
#
#   bash scripts/remote_sync_leviathan.sh          # code + checkpoint + datasets
#   WHAT=code bash scripts/remote_sync_leviathan.sh   # only the repo (quick re-sync)

KEY="${KEY:?ssh private key of the remote machine}"
REMOTE="${REMOTE:?user@host of the remote machine}"
RBASE="${RBASE:?base directory on the remote machine}"
SRC_ROOT="${SRC_ROOT:-/media/ilias/DATA/ilias}"
WHAT="${WHAT:-all}"
SSH="ssh -i $KEY -o BatchMode=yes"
RS=(rsync -a --partial --info=stats1,progress2 -e "$SSH")

$SSH "$REMOTE" "mkdir -p $RBASE/AMROD $RBASE/AMROD_data/datasets/ACDC/rgb_anon $RBASE/AMROD_data/datasets/cityscapes/gtFine $RBASE/AMROD_data/output/panoptic_fpn_R50_cityscapes_segw1 $RBASE/envs"

echo "== code"
{ git -C "$SRC_ROOT/AMROD" rev-parse HEAD; git -C "$SRC_ROOT/AMROD" status --porcelain | head -50; } > /tmp/PROVENANCE.txt

"${RS[@]}" --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=nohup.out \
  --exclude='*.zip' --exclude='*.bak*' --exclude='*.pre_*' --exclude=mtl_papers --exclude=papers \
  "$SRC_ROOT/AMROD/" "$REMOTE:$RBASE/AMROD/"
"${RS[@]}" /tmp/PROVENANCE.txt "$REMOTE:$RBASE/AMROD/"
[[ "$WHAT" == code ]] && exit 0

echo "== checkpoint (final weights only)"
"${RS[@]}" "$SRC_ROOT/amrod_output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" \
  "$SRC_ROOT/amrod_output/panoptic_fpn_R50_cityscapes_segw1/config.yaml" \
  "$REMOTE:$RBASE/AMROD_data/output/panoptic_fpn_R50_cityscapes_segw1/"

echo "== ACDC (4 conditions, train split used by the benchmark)"
for w in fog night rain snow; do
  $SSH "$REMOTE" "mkdir -p $RBASE/AMROD_data/datasets/ACDC/rgb_anon/$w"
  "${RS[@]}" "$SRC_ROOT/acdc/rgb_anon/$w/train" "$REMOTE:$RBASE/AMROD_data/datasets/ACDC/rgb_anon/$w/"
done
"${RS[@]}" "$SRC_ROOT/acdc/gt_detection" "$SRC_ROOT/acdc/gt" "$REMOTE:$RBASE/AMROD_data/datasets/ACDC/"

echo "== Cityscapes detection GT + val labels"
"${RS[@]}" -L "$SRC_ROOT/cityscapes/instancesonly_filtered_gtFine_val.json" \
  "$REMOTE:$RBASE/AMROD_data/datasets/cityscapes/"
$SSH "$REMOTE" "mkdir -p $RBASE/AMROD_data/datasets/cityscapes/annotations && cd $RBASE/AMROD_data/datasets/cityscapes/annotations && ln -sf ../instancesonly_filtered_gtFine_val.json ."
"${RS[@]}" "$SRC_ROOT/cityscapes/gtFine/val" "$REMOTE:$RBASE/AMROD_data/datasets/cityscapes/gtFine/"

echo "== Cityscapes-C (12 corruptions, val)"
for c in defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression; do
  $SSH "$REMOTE" "mkdir -p $RBASE/AMROD_data/datasets/$c/leftImg8bit"
  "${RS[@]}" "$SRC_ROOT/cityscapes_c/$c/leftImg8bit/val" "$REMOTE:$RBASE/AMROD_data/datasets/$c/leftImg8bit/"
done
echo "SYNC DONE"
