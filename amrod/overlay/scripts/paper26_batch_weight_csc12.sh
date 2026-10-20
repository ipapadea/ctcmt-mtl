#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 — CT-CR weight sensitivity on CSC12, seed 0.
# Usage:
#   DRY_RUN=1 bash paper26_batch_weight_csc12.sh 0
#   bash paper26_batch_weight_csc12.sh 0
# Optional: FORCE=1 to deliberately rerun/overwrite a paper26_* experiment.

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
RUNNER="${ROOT}/scripts/run_mixed_lt_local.sh"
CFG="detectron2/configs/Cityscapes/newsrc_csc_ctcr.yaml"
SEED=0
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog \
       brightness contrast elastic_transform pixelate jpeg_compression)
STREAM="("
for c in "${CSC12[@]}"; do STREAM+="\"${c}_mtl\","; done
STREAM="${STREAM%,})"

weights=(0.0 0.1 0.3 0.5 1.0)
tags=(0p0 0p1 0p3 0p5 1p0)

[[ -x "$RUNNER" || -f "$RUNNER" ]] || { echo "ERROR: missing $RUNNER" >&2; exit 2; }
[[ -f "$ROOT/$CFG" ]] || { echo "ERROR: missing $ROOT/$CFG" >&2; exit 2; }
[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || {
  echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }

p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

run_one() {
  local w="$1" tag="$2" rc
  local exp="paper26_weight_csc12_w${tag}_s${SEED}"
  local log="$HOST_OUT/logs/${exp}.log"
  local out="$HOST_OUT/ctta_acdc/${exp}"
  local meta="$HOST_OUT/paper26_meta/${exp}.txt"

  echo "[GPU $GPU] $exp  weight=$w  seed=$SEED  cfg=$CFG"
  p26_gate "$exp" || return 0
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  STREAM='$STREAM'"
    echo "  EXTRA_OPTS='SOLVER.CTCMT_WEIGHT_CTCR $w'"
    return 0
  fi

  cat > "$meta" <<META
experiment=$exp
protocol=CSC12
seed=$SEED
gpu=$GPU
config=$CFG
ctcr_weight=$w
ctcr_mode=soft_seg_global
ctcr_floor=0.2
git_sha=$sha
stream=$STREAM
expected_evals=12
start_time=$(date -Is)
META

  set +e
  STREAM="$STREAM" EXTRA_OPTS="SOLVER.CTCMT_WEIGHT_CTCR $w" \
    bash "$RUNNER" "$GPU" "$exp" "$CFG" "$SEED"
  rc=$?
  set -e
  p26_finish "$exp" "$rc" 12 "$log" "$meta" || exit 1
}

for i in "${!weights[@]}"; do run_one "${weights[i]}" "${tags[i]}"; done

echo "DONE: CSC12 CT-CR weight sweep"
