#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 — CT-CR weight sensitivity on ACDC4, seed 0.
# Run after CSC12 sweep if budget allows.
# Usage: DRY_RUN=1 bash paper26_batch_weight_acdc4.sh 0

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
RUNNER="${ROOT}/scripts/run_ctcr_acdc_local.sh"
CFG="detectron2/configs/Cityscapes/newsrc_acdc_ctcr.yaml"
SEED=0
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
STREAM='("acdc_fog_mtl","acdc_night_mtl","acdc_rain_mtl","acdc_snow_mtl")'
weights=(0.0 0.1 0.3 0.5 1.0)
tags=(0p0 0p1 0p3 0p5 1p0)

[[ -f "$ROOT/$CFG" ]] || { echo "ERROR: missing $ROOT/$CFG" >&2; exit 2; }
[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

for i in "${!weights[@]}"; do
  w="${weights[i]}"; tag="${tags[i]}"; exp="paper26_weight_acdc4_w${tag}_s${SEED}"
  log="$HOST_OUT/logs/${exp}.log"; out="$HOST_OUT/ctta_acdc/${exp}"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
  echo "[GPU $GPU] $exp  weight=$w  seed=$SEED  cfg=$CFG"
  p26_gate "$exp" || continue
  if [[ "$DRY_RUN" == 1 ]]; then echo "  STREAM='$STREAM'"; echo "  EXTRA_OPTS='SOLVER.CTCMT_WEIGHT_CTCR $w'"; continue; fi
  cat > "$meta" <<META
experiment=$exp
protocol=ACDC4
seed=$SEED
gpu=$GPU
config=$CFG
ctcr_weight=$w
ctcr_mode=soft_seg_global
ctcr_floor=0.2
git_sha=$sha
stream=$STREAM
expected_evals=4
start_time=$(date -Is)
META
  set +e
  STREAM="$STREAM" EXTRA_OPTS="SOLVER.CTCMT_WEIGHT_CTCR $w" bash "$RUNNER" "$GPU" "$exp" "$CFG" "$SEED"
  rc=$?
  set -e
  p26_finish "$exp" "$rc" 4 "$log" "$meta" || exit 1
done

echo "DONE: ACDC4 CT-CR weight sweep"
