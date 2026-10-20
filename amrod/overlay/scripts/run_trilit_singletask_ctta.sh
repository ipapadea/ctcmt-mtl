#!/usr/bin/env bash
# Auto-launch single-task CTTA once det/seg training finishes.
#
# Waits for both training logs to show "[done]", then runs:
#   GPU 0  AMROD on det-only TriLiteNet  (1-round + 10-round)
#   GPU 1  CoTTA on seg-only TriLiteNet  (1-round + 10-round)
#
# Usage (run inside tmux so it survives terminal close):
#   tmux new-session -d -s trilit-singletask \
#     "bash /home/ilias/AMROD/scripts/run_trilit_singletask_ctta.sh \
#      2>&1 | tee /tmp/trilit_singletask_launcher.log"

set -euo pipefail

DET_LOG="/tmp/trilit_det_resume.log"
SEG_LOG="/tmp/trilit_seg_resume.log"
TRILIT="/home/ilias/TriLiteNet"
ACDC="/data/ilias/acdc"

CKPT_DET="${TRILIT}/runs/cityscapes_det_base_native/ckpt_last.pth"
CKPT_SEG="${TRILIT}/runs/cityscapes_seg_base_native/ckpt_best_seg.pth"

_wait_done() {
    local log="$1" label="$2"
    echo "[wait] Polling ${label} training (${log}) ..."
    while ! grep -q "\[done\]" "$log" 2>/dev/null; do
        epoch=$(grep -oP "epoch\s+\K[0-9]+" "$log" 2>/dev/null | tail -1 || echo "?")
        echo "[wait] ${label} still training (last epoch=${epoch}) — sleeping 5 min ..."
        sleep 300
    done
    echo "[wait] ${label} training DONE."
}

_docker() {
    local gpu="$1" name="$2" ckpt="$3" method="$4" repeats="$5"; shift 5
    echo "[run] ${name} (GPU ${gpu}, method=${method}, repeats=${repeats}) ..."
    docker run --rm --gpus "\"device=${gpu}\"" --shm-size=8g \
      -v "${TRILIT}:/workspace" \
      -v "${ACDC}:/data/ilias/acdc:ro" \
      trilitenet:latest \
      python tools/adapt_ctta.py \
        --ckpt "${ckpt}" \
        --method "${method}" \
        --num-seg-classes 19 --num-det-classes 8 \
        --num-repeats "${repeats}" --device 0 \
        "$@" \
      2>&1 | tee "/tmp/trilit_${name}.log"
    echo "[run] ${name} DONE."
}

# ---- Wait for both training runs ----
_wait_done "$DET_LOG" "det-only"
_wait_done "$SEG_LOG" "seg-only"

echo ""
echo "=== Both single-task models ready. Launching CTTA experiments. ==="
echo ""

# ---- AMROD on det-only source (GPU 0) ----
# 1-round short task
_docker 0 "amrod_det_x1" "$CKPT_DET" "amrod_mtl" 1 \
  --amrod-det-weight 1.0 --amrod-seg-weight 0.0 \
  --run-name "cityscapes_det_base_native__amrod__x1" &
PID_AMROD=$!

# ---- CoTTA on seg-only source (GPU 1) ----
# 1-round short task
_docker 1 "cotta_seg_x1" "$CKPT_SEG" "cotta" 1 \
  --run-name "cityscapes_seg_base_native__cotta__x1" &
PID_COTTA=$!

wait $PID_AMROD $PID_COTTA

echo ""
echo "=== Short-task (1 round) complete. Starting long-term (10 rounds). ==="
echo ""

# ---- 10-round long-term ----
_docker 0 "amrod_det_x10" "$CKPT_DET" "amrod_mtl" 10 \
  --amrod-det-weight 1.0 --amrod-seg-weight 0.0 \
  --run-name "cityscapes_det_base_native__amrod__x10" &
PID_AMROD10=$!

_docker 1 "cotta_seg_x10" "$CKPT_SEG" "cotta" 10 \
  --run-name "cityscapes_seg_base_native__cotta__x10" &
PID_COTTA10=$!

wait $PID_AMROD10 $PID_COTTA10

# ---- Source-only baselines for completeness ----
echo ""
echo "=== Source-only baselines ==="
_docker 0 "source_det_x1" "$CKPT_DET" "amrod_mtl" 1 \
  --source-only --run-name "cityscapes_det_base_native__source__x1"
_docker 1 "source_seg_x1" "$CKPT_SEG" "cotta" 1 \
  --source-only --run-name "cityscapes_seg_base_native__source__x1"

echo ""
echo "=== All experiments done. Results summary: ==="
for f in amrod_det_x1 cotta_seg_x1 amrod_det_x10 cotta_seg_x10 source_det_x1 source_seg_x1; do
    echo "--- ${f} ---"
    grep -E "^\[global\]|^  fog|^  night|^  rain|^  snow" \
        "/tmp/trilit_${f}.log" 2>/dev/null | head -5 || echo "(no results)"
done
