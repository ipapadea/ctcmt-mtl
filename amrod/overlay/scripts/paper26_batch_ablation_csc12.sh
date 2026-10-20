#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 — clean 3-seed component ablation on CSC12.
# Uses the repository's established multi-seed convention 0,42,123.
# 5 variants x 3 seeds = 15 runs, sequentially on one GPU.
# Usage:
#   DRY_RUN=1 bash paper26_batch_ablation_csc12.sh 1
#   bash paper26_batch_ablation_csc12.sh 1
# Optional: VARIANTS='ctcl both all' or SEEDS='42 123'.

GPU="${1:-1}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
RUNNER="${ROOT}/scripts/run_csc12_local.sh"
CFGROOT="detectron2/configs/Cityscapes"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
SEEDS_STR="${SEEDS:-0 42 123}"
VARIANTS_STR="${VARIANTS:-base ctcr ctcl both all}"
read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a VARIANT_ARR <<< "$VARIANTS_STR"

cfg_for() {
  case "$1" in
    base) echo "$CFGROOT/newsrc_csc_base.yaml" ;;
    ctcr) echo "$CFGROOT/newsrc_csc_ctcr.yaml" ;;
    ctcl) echo "$CFGROOT/newsrc_csc_ctcl.yaml" ;;
    both) echo "$CFGROOT/newsrc_csc_ctcl_ctcr.yaml" ;;
    all)  echo "$CFGROOT/newsrc_csc_ctcl_ctcr_ctpv.yaml" ;;
    *) echo "ERROR: unknown variant $1" >&2; return 2 ;;
  esac
}

[[ -f "$RUNNER" ]] || { echo "ERROR: missing $RUNNER" >&2; exit 2; }
[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }
for v in "${VARIANT_ARR[@]}"; do c="$(cfg_for "$v")"; [[ -f "$ROOT/$c" ]] || { echo "ERROR: missing $ROOT/$c" >&2; exit 2; }; done
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

for seed in "${SEED_ARR[@]}"; do
  for v in "${VARIANT_ARR[@]}"; do
    cfg="$(cfg_for "$v")"
    exp="paper26_ablate_csc12_${v}_s${seed}"
    log="$HOST_OUT/logs/${exp}.log"; out="$HOST_OUT/ctta_acdc/${exp}"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
    echo "[GPU $GPU] $exp  seed=$seed  cfg=$cfg"
    p26_gate "$exp" || continue
    [[ "$DRY_RUN" == 1 ]] && continue
    cat > "$meta" <<META
experiment=$exp
protocol=CSC12
variant=$v
seed=$seed
gpu=$GPU
config=$cfg
git_sha=$sha
expected_evals=12
start_time=$(date -Is)
META
    set +e
    bash "$RUNNER" "$GPU" "$exp" "$cfg" "$seed"
    rc=$?
    set -e
    p26_finish "$exp" "$rc" 12 "$log" "$meta" || exit 1
  done
done

echo "DONE: CSC12 multi-seed component ablation"
