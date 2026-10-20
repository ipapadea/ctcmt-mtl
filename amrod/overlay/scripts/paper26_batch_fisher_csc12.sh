#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 — Fisher experimental variants on CSC12, seed 0 only.
# Variant A (random restoration) is the existing canonical run and is NOT rerun.
# Usage: DRY_RUN=1 bash paper26_batch_fisher_csc12.sh GPU ; VARIANTS='restore xtask' to subset.

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
RUNNER="${ROOT}/scripts/run_csc12_local.sh"
CFGROOT="detectron2/configs/Cityscapes"
SEED=0
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
read -r -a VARIANT_ARR <<< "${VARIANTS:-restore norestore xtask}"

cfg_for() {
  case "$1" in
    restore)   echo "$CFGROOT/newsrc_csc_ctcr_fisherrestore.yaml" ;;
    norestore) echo "$CFGROOT/newsrc_csc_ctcr_norestore.yaml" ;;
    xtask)     echo "$CFGROOT/newsrc_csc_ctcr_fisherxtask.yaml" ;;
    *) echo "ERROR: unknown variant $1" >&2; return 2 ;;
  esac
}
name_for() {
  case "$1" in
    restore)   echo "paper26_fisher_restore_csc12_s${SEED}" ;;
    norestore) echo "paper26_fisher_norestore_csc12_s${SEED}" ;;
    xtask)     echo "paper26_fisher_xtask_csc12_s${SEED}" ;;
  esac
}

[[ -f "$RUNNER" ]] || { echo "ERROR: missing $RUNNER" >&2; exit 2; }
[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical checkpoint missing" >&2; exit 2; }
for v in "${VARIANT_ARR[@]}"; do c="$(cfg_for "$v")"; [[ -f "$ROOT/$c" ]] || { echo "ERROR: missing $ROOT/$c" >&2; exit 2; }; done
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

for v in "${VARIANT_ARR[@]}"; do
  cfg="$(cfg_for "$v")"; exp="$(name_for "$v")"
  log="$HOST_OUT/logs/${exp}.log"; out="$HOST_OUT/ctta_acdc/${exp}"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
  echo "[GPU $GPU] $exp  variant=$v  seed=$SEED  cfg=$cfg"
  p26_gate "$exp" || continue
  [[ "$DRY_RUN" == 1 ]] && continue
  cat > "$meta" <<META
experiment=$exp
protocol=CSC12
variant=$v
seed=$SEED
gpu=$GPU
config=$cfg
git_sha=$sha (working tree has uncommitted Fisher changes)
expected_evals=12
start_time=$(date -Is)
META
  set +e
  bash "$RUNNER" "$GPU" "$exp" "$cfg" "$SEED"
  rc=$?
  set -e
  p26_finish "$exp" "$rc" 12 "$log" "$meta" || exit 1
done
echo "DONE: Fisher variants CSC12"
