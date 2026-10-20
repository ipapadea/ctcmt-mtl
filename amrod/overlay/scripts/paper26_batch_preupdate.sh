#!/usr/bin/env bash
set -Eeuo pipefail

# Paper 2026 - evaluation-protocol control: CTCMT and Base with the PRE-update
# teacher prediction (SOLVER.CTCMT_PREDICT_BEFORE_UPDATE=True, predict-then-adapt),
# to compare fairly with AMROD / CTAOD / OBAO / DAT-BF, which predict before the update.
# Everything else (config, seeds, streams, checkpoint) is identical to the reported runs.
#
# Defaults: short-term ACDC4 + CSC12, variants ctcmt (= headline lambda1) and base, seeds 0 42 123
#           -> 2 protocols x 2 variants x 3 seeds = 12 runs on one GPU.
# Usage:
#   DRY_RUN=1 bash paper26_batch_preupdate.sh 0
#   bash paper26_batch_preupdate.sh 0
#   GPU 1 in parallel: PROTOCOLS='csc12' bash paper26_batch_preupdate.sh 1
# Optional: PROTOCOLS='acdc4 csc12 acdcLT cscLT'  VARIANTS='ctcmt base'  SEEDS='0 42 123'
#           (long-term runs are slow; use SEEDS=0 for them)

GPU="${1:-0}"
ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CFGROOT="detectron2/configs/Cityscapes"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
source "$(dirname "${BASH_SOURCE[0]}")/paper26_common.sh"
read -r -a PROTO_ARR <<< "${PROTOCOLS:-acdc4 csc12}"
read -r -a VARIANT_ARR <<< "${VARIANTS:-ctcmt base}"
read -r -a SEED_ARR <<< "${SEEDS:-0 42 123}"

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
ACDC4=(acdc_fog acdc_night acdc_rain acdc_snow)

make_stream() {  # make_stream REPEATS item...
  local reps="$1"; shift
  local s="(" r c
  for ((r = 0; r < reps; r++)); do for c in "$@"; do s+="\"${c}_mtl\","; done; done
  echo "${s%,})"
}

# Sets RUNNER CFG STREAM EVALS for protocol $1 / variant $2.
resolve() {
  local proto="$1" v="$2" dom
  case "$proto" in
    acdc4)  dom=acdc; RUNNER="$ROOT/scripts/run_ctcr_acdc_local.sh"; STREAM="$(make_stream 1 "${ACDC4[@]}")"; EVALS=4 ;;
    acdcLT) dom=acdc; RUNNER="$ROOT/scripts/run_ctcr_acdc_local.sh"; STREAM="$(make_stream 10 "${ACDC4[@]}")"; EVALS=40 ;;
    csc12)  dom=csc;  RUNNER="$ROOT/scripts/run_mixed_lt_local.sh";  STREAM="$(make_stream 1 "${CSC12[@]}")"; EVALS=12 ;;
    cscLT)  dom=csc;  RUNNER="$ROOT/scripts/run_mixed_lt_local.sh";  STREAM="$(make_stream 10 fog motion_blur snow brightness defocus_blur)"; EVALS=50 ;;
    *) echo "ERROR: unknown protocol $proto" >&2; return 2 ;;
  esac
  case "$v" in
    ctcmt) CFG="$CFGROOT/newsrc_${dom}_ctcr_lambda1.yaml" ;;  # headline config
    base)  CFG="$CFGROOT/newsrc_${dom}_base.yaml" ;;
    *) echo "ERROR: unknown variant $v" >&2; return 2 ;;
  esac
}

[[ -f "$HOST_OUT/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || { echo "ERROR: canonical new-source checkpoint missing" >&2; exit 2; }
grep -q 'CTCMT_PREDICT_BEFORE_UPDATE' "$ROOT/detectron2/detectron2/config/defaults.py" || { echo "ERROR: CTCMT_PREDICT_BEFORE_UPDATE not defined in defaults.py" >&2; exit 2; }
for p in "${PROTO_ARR[@]}"; do for v in "${VARIANT_ARR[@]}"; do
  resolve "$p" "$v" || exit 2
  [[ -f "$RUNNER" && -f "$ROOT/$CFG" ]] || { echo "ERROR: missing $RUNNER or $ROOT/$CFG" >&2; exit 2; }
done; done
p26_init
sha="$(git -C "$ROOT" rev-parse HEAD)"

for seed in "${SEED_ARR[@]}"; do
  for p in "${PROTO_ARR[@]}"; do
    for v in "${VARIANT_ARR[@]}"; do
      resolve "$p" "$v"
      exp="paper26_preupd_${p}_${v}_s${seed}"
      log="$HOST_OUT/logs/${exp}.log"; meta="$HOST_OUT/paper26_meta/${exp}.txt"
      echo "[GPU $GPU] $exp  protocol=$p  variant=$v  seed=$seed  cfg=$CFG"
      p26_gate "$exp" || continue
      if [[ "$DRY_RUN" == 1 ]]; then
        echo "  RUNNER=$(basename "$RUNNER")  EVALS=$EVALS"
        echo "  EXTRA_OPTS='SOLVER.CTCMT_PREDICT_BEFORE_UPDATE True'"
        continue
      fi
      cat > "$meta" <<META
experiment=$exp
protocol=$p
variant=$v
seed=$seed
gpu=$GPU
config=$CFG
predict_before_update=True
git_sha=$sha
git_dirty=$(git -C "$ROOT" status --porcelain -- detectron2/detectron2 | wc -l)
expected_evals=$EVALS
start_time=$(date -Is)
META
      set +e
      STREAM="$STREAM" EXTRA_OPTS="SOLVER.CTCMT_PREDICT_BEFORE_UPDATE True" \
        bash "$RUNNER" "$GPU" "$exp" "$CFG" "$seed"
      rc=$?
      set -e
      p26_finish "$exp" "$rc" "$EVALS" "$log" "$meta" || exit 1
    done
  done
done

echo "DONE: pre-update prediction control"
