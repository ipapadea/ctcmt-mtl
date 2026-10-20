#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 - single-task baseline: MT + object-level CL + SR (Moraiti et al., EJAI 2025),
# same-source port onto the Panoptic-FPN R50 detection branch (meta-arch MTCLSR).
# Hyper-parameters are the paper's (5 updates, 2 views, alpha 0.99965, p 0.01, ...);
# see detectron2/configs/Cityscapes/mtclsr_pfn_R_50_ACDC.yaml.
#
# Defaults: all four protocols, seed 0.
# Usage:
#   DRY_RUN=1 bash paper26_batch_mtclsr.sh 0
#   PROTOCOLS='acdc4 csc12 acdcLT' bash paper26_batch_mtclsr.sh 0
#   PROTOCOLS='cscLT' bash paper26_batch_mtclsr.sh 1
# Remote (no docker): REMOTE_CONDA=1 HOST_REPO=... HOST_OUT=... bash paper26_batch_mtclsr.sh 0
# Optional: SEEDS='0 42 123'  PREDICT=post (extra teacher pass after the last update)

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CFGROOT="detectron2/configs/Cityscapes"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
PREDICT="${PREDICT:-last_step}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
read -r -a PROTO_ARR <<< "${PROTOCOLS:-acdc4 csc12 acdcLT cscLT}"
read -r -a SEED_ARR <<< "${SEEDS:-0}"

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
CSC5=(fog motion_blur snow brightness defocus_blur)
ACDC4=(acdc_fog acdc_night acdc_rain acdc_snow)

make_stream() {  # make_stream REPEATS item...  (detection-only dataset names, no _mtl suffix)
  local reps="$1"; shift
  local s="(" r c
  for ((r = 0; r < reps; r++)); do for c in "$@"; do s+="\"${c}\","; done; done
  echo "${s%,})"
}

# Sets RUNNER CFG STREAM EVALS for protocol $1.
resolve() {
  case "$1" in
    acdc4)  RUNNER="$ROOT/scripts/run_ctcr_acdc_local.sh"; CFG="$CFGROOT/mtclsr_pfn_R_50_ACDC.yaml"; STREAM="$(make_stream 1 "${ACDC4[@]}")"; EVALS=4 ;;
    acdcLT) RUNNER="$ROOT/scripts/run_ctcr_acdc_local.sh"; CFG="$CFGROOT/mtclsr_pfn_R_50_ACDC.yaml"; STREAM="$(make_stream 10 "${ACDC4[@]}")"; EVALS=40 ;;
    csc12)  RUNNER="$ROOT/scripts/run_mixed_lt_local.sh";  CFG="$CFGROOT/mtclsr_pfn_R_50_CS_C.yaml"; STREAM="$(make_stream 1 "${CSC12[@]}")"; EVALS=12 ;;
    cscLT)  RUNNER="$ROOT/scripts/run_mixed_lt_local.sh";  CFG="$CFGROOT/mtclsr_pfn_R_50_CS_C.yaml"; STREAM="$(make_stream 10 "${CSC5[@]}")"; EVALS=50 ;;
    *) echo "ERROR: unknown protocol $1" >&2; return 2 ;;
  esac
  # Remote machine without docker: same interface, conda environment instead of the image.
  [[ "${REMOTE_CONDA:-0}" == 1 ]] && RUNNER="$ROOT/scripts/run_remote_conda.sh"
  return 0
}

[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }
[[ -f "$ROOT/detectron2/detectron2/modeling/meta_arch/mtclsr.py" ]] || { echo "ERROR: meta-arch mtclsr.py missing" >&2; exit 2; }
for p in "${PROTO_ARR[@]}"; do
  resolve "$p" || exit 2
  [[ -f "$RUNNER" && -f "$ROOT/$CFG" ]] || { echo "ERROR: missing $RUNNER or $ROOT/$CFG" >&2; exit 2; }
done
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || cat "$ROOT/PROVENANCE.txt" 2>/dev/null | head -1 || echo unknown)"

for seed in "${SEED_ARR[@]}"; do
  for p in "${PROTO_ARR[@]}"; do
    resolve "$p"
    exp="paper26_mtclsr_${p}_s${seed}"
    log="$HOST_OUT/logs/${exp}.log"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
    echo "[GPU $GPU] $exp  protocol=$p  seed=$seed  cfg=$CFG  predict=$PREDICT"
    p26_gate "$exp" || continue
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "  RUNNER=$(basename "$RUNNER")  EVALS=$EVALS"
      echo "  EXTRA_OPTS='TEST.MTCLSR.PREDICT $PREDICT'"
      continue
    fi
    cat > "$meta" <<META
experiment=$exp
method=MTCLSR (Moraiti et al. port, Panoptic-FPN detection branch)
protocol=$p
seed=$seed
gpu=$GPU
config=$CFG
predict=$PREDICT
git_sha=$sha
git_dirty=$(git -C "$ROOT" status --porcelain -- detectron2/detectron2 detectron2/configs 2>/dev/null | wc -l)
expected_evals=$EVALS
start_time=$(date -Is)
META
    set +e
    STREAM="$STREAM" EXTRA_OPTS="TEST.MTCLSR.PREDICT $PREDICT" \
      bash "$RUNNER" "$GPU" "$exp" "$CFG" "$seed"
    rc=$?
    set -e
    p26_finish "$exp" "$rc" "$EVALS" "$log" "$meta" || exit 1
  done
done

echo "DONE: MTCLSR single-task baseline"
