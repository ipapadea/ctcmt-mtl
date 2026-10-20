#!/usr/bin/env bash
set -euo pipefail

# Non-docker equivalent of run_ctcr_acdc_local.sh / run_mixed_lt_local.sh for the remote
# machine ("leviathan").  Same positional interface, so paper26_batch_mtclsr.sh can use it:
#
#   STREAM='("acdc_fog",)' bash scripts/run_remote_conda.sh GPU EXP_NAME CONFIG [SEED]
#
# Paths come from env vars (defaults = layout created by remote_sync_leviathan.sh).

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="${HOST_REPO:-/mnt/data_storage/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/mnt/data_storage/ilias/AMROD_data/output}"
DATA_ROOT="${DATA_ROOT:-/mnt/data_storage/ilias/AMROD_data/datasets}"
ENV="${ENV:-/mnt/data_storage/ilias/envs/amrod}"
EXTRA_OPTS="${EXTRA_OPTS:-}"
STREAM="${STREAM:?STREAM required (python tuple of dataset names)}"

LOG="${HOST_OUT}/logs/${EXP}.log"
OUT="${HOST_OUT}/ctta_acdc/${EXP}"
CKPT="${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
mkdir -p "$(dirname "$LOG")"
[[ -f "$CKPT" ]] || { echo "ERROR: missing $CKPT" >&2; exit 2; }
[[ -d "$DATA_ROOT/ACDC/rgb_anon" && -d "$DATA_ROOT/cityscapes" ]] || { echo "ERROR: datasets missing under $DATA_ROOT" >&2; exit 2; }

echo "======================================================================"
echo "REMOTE RUN  EXP: ${EXP}   GPU: ${GPU}   SEED: ${SEED}"
echo "CONFIG : ${CFG}"
echo "======================================================================"

rm -rf "$OUT"
export PATH="$ENV/bin:$PATH"   # no `conda activate`: its gcc hooks break under set -u
cd "$HOST_REPO"

# the system libstdc++ (Ubuntu 18.04) is older than the one the compiled detectron2 ops need
LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}" \
CUDA_VISIBLE_DEVICES="$GPU" \
NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 \
DETECTRON2_DATASETS="$DATA_ROOT" \
PYTHONPATH="$HOST_REPO/detectron2" \
"$ENV/bin/python" detectron2/tools/train_net.py \
  --config-file "$CFG" \
  --eval-only \
  --num-gpus 1 \
  OUTPUT_DIR "$OUT" \
  MODEL.WEIGHTS "$CKPT" \
  SEED "$SEED" \
  DATASETS.TEST "$STREAM" $EXTRA_OPTS 2>&1 | tee "$LOG"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
