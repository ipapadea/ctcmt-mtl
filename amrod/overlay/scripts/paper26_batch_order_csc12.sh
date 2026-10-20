#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 — CSC12 stream-order robustness, Base vs Ours.
# Default is exploratory seed 0. Later: SEEDS='0 42 123'.
# Orders: canonical, reverse, deterministic random permutation (seed 2026).
# Usage: DRY_RUN=1 bash paper26_batch_order_csc12.sh 0

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
RUNNER="${ROOT}/scripts/run_mixed_lt_local.sh"
CFGROOT="detectron2/configs/Cityscapes"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
SEEDS_STR="${SEEDS:-0}"
read -r -a SEED_ARR <<< "$SEEDS_STR"

canonical=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
reverse=(jpeg_compression pixelate elastic_transform contrast brightness fog frost snow zoom_blur motion_blur glass_blur defocus_blur)
random2026=(defocus_blur fog motion_blur zoom_blur pixelate snow brightness jpeg_compression elastic_transform contrast frost glass_blur)

stream_from_array() {
  local -n arr=$1; local s="(" c
  for c in "${arr[@]}"; do s+="\"${c}_mtl\","; done
  printf '%s' "${s%,})"
}

cfg_for() { [[ "$1" == base ]] && echo "$CFGROOT/newsrc_csc_base.yaml" || echo "$CFGROOT/newsrc_csc_ctcr.yaml"; }
[[ -f "$RUNNER" ]] || { echo "ERROR: missing $RUNNER" >&2; exit 2; }
for c in base ours; do [[ -f "$ROOT/$(cfg_for "$c")" ]] || { echo "ERROR: missing config for $c" >&2; exit 2; }; done
[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

for seed in "${SEED_ARR[@]}"; do
  for order in canonical reverse random2026; do
    stream="$(stream_from_array "$order")"
    for method in base ours; do
      cfg="$(cfg_for "$method")"
      exp="paper26_order_csc12_${order}_${method}_s${seed}"
      log="$HOST_OUT/logs/${exp}.log"; out="$HOST_OUT/ctta_acdc/${exp}"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
      echo "[GPU $GPU] $exp  seed=$seed  cfg=$cfg"
      p26_gate "$exp" || continue
      if [[ "$DRY_RUN" == 1 ]]; then echo "  STREAM=$stream"; continue; fi
      cat > "$meta" <<META
experiment=$exp
protocol=CSC12-order
method=$method
order=$order
seed=$seed
gpu=$GPU
config=$cfg
git_sha=$sha
stream=$stream
random_order_seed=2026
expected_evals=12
start_time=$(date -Is)
META
      set +e
      STREAM="$stream" EXTRA_OPTS="" bash "$RUNNER" "$GPU" "$exp" "$cfg" "$seed"
      rc=$?
      set -e
      p26_finish "$exp" "$rc" 12 "$log" "$meta" || exit 1
    done
  done
done

echo "DONE: CSC12 order robustness"
