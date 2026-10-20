#!/usr/bin/env bash
# XAI-guided pseudo-label filter (Option 1) sweep, GPU 1 only.
#
# Structure (12 runs, sequential, ~30 min each = ~6 h wall time):
#   phase1 (3 runs): XAI technique comparison
#     method in {eigencam, featnorm, gradcam}  x  thresh=0.02 drop  TTBBR=off
#     -> isolates the XAI signal from TT-BBR
#   phase2 (3 runs): composability with TT-BBR (iou=0.7, drop=False)
#     method in {eigencam, featnorm, gradcam}  x  thresh=0.02 drop  TTBBR=on
#     -> can we stack XAI + TT-BBR without anti-synergy?
#   phase3 (6 runs): threshold sweep at BEST method (default eigencam), TTBBR=off
#     thresh in {0.005, 0.01, 0.03, 0.05, 0.08, 0.12}
#     -> where is the sweet spot on the drop threshold?
#
# Note (2026-07-28 recalibration): initial threshold=0.30 dropped ALL boxes
# because real FR-CNN 7x7 pooled features on Cityscapes-C produce
# normalized-entropy concentration values in the 0.02-0.15 range. Sweep
# retuned to the observed regime.
#
# Idempotent: skips runs whose stdout.log already contains AP50_ALL_mean.
# Auto-tmux self-relaunch (like sweep_ttbbr_long.sh) so the invoking shell
# is not held synchronously.
#
# Usage:
#   ./scripts/sweep_xai_short.sh                        # all phases
#   ./scripts/sweep_xai_short.sh phase1                 # method comparison only
#   ./scripts/sweep_xai_short.sh phase2                 # +TT-BBR composability
#   ./scripts/sweep_xai_short.sh phase3 [best_method]   # threshold sweep at best method
set -euo pipefail

# ---- Auto-tmux self-relaunch guard --------------------------------------
# On first invocation from a regular shell, this script re-fires itself
# inside a persistent tmux session and returns immediately. The dispatch
# loop runs in the `dispatcher` window; each XAI training run is spawned
# in its own tmux window under the same session.
if [ "${XAI_SHORT_IN_TMUX:-0}" != "1" ]; then
    export XAI_SHORT_IN_TMUX=1
    tmux has-session -t xai-short 2>/dev/null \
        || tmux new-session -d -s xai-short -n control
    tmux new-window -t xai-short -n dispatcher \
        "XAI_SHORT_IN_TMUX=1 bash '$0' $* 2>&1 | tee /tmp/xai_short_dispatcher.log; exec bash -l"
    echo "[sweep-xai] dispatcher launched in tmux session 'xai-short'."
    echo "[sweep-xai] attach with:   tmux attach -t xai-short"
    echo "[sweep-xai] tail log with: tail -f /tmp/xai_short_dispatcher.log"
    exit 0
fi

PHASE="${1:-all}"
BEST_METHOD="${2:-eigencam}"

SWEEP_ROOT=/data/vgcmt/work_dirs/xai_short_sweep
mkdir -p "$SWEEP_ROOT"

# Single-GPU sequential dispatch on GPU 1 (GPUs 2/3 are held by ttbbr sweeps).
GPU_LIST=(1)
NUM_GPUS=${#GPU_LIST[@]}

CKPT=/data/vgcmt/models/amrod_d2/cityscapes_train_final.pth
DATASETS=/data/vgcmt/datasets/cityscapes_c_amrod
CFG=cfg_cityscapes_c_short_xai.yaml

# Job queue: each entry is "name method thresh mode ttbbr_on"
declare -a JOBS=()

# ---- Phase 1: XAI technique comparison, no TT-BBR --------------------
# Very low default threshold (0.02): first observations showed real
# concentration percentiles hover well below 0.1 on FR-CNN 7x7 features.
# Phase 3 sweeps thresholds more broadly.
if [[ "$PHASE" == "phase1" || "$PHASE" == "all" ]]; then
    for m in eigencam featnorm gradcam; do
        name="xai${m}_th0.02_drop_ttbbr0"
        JOBS+=("$name $m 0.02 drop 0")
    done
fi

# ---- Phase 2: composability with TT-BBR (iou=0.7, drop=False) --------
if [[ "$PHASE" == "phase2" || "$PHASE" == "all" ]]; then
    for m in eigencam featnorm gradcam; do
        name="xai${m}_th0.02_drop_ttbbr1"
        JOBS+=("$name $m 0.02 drop 1")
    done
fi

# ---- Phase 3: threshold sweep at BEST method, TT-BBR off -------------
# Six-point sweep covering both the aggressive-drop and near-noop regimes.
if [[ "$PHASE" == "phase3" || "$PHASE" == "all" ]]; then
    for th in 0.005 0.01 0.03 0.05 0.08 0.12; do
        name="xai${BEST_METHOD}_th${th}_drop_ttbbr0"
        JOBS+=("$name $BEST_METHOD $th drop 0")
    done
fi

echo "[sweep-xai] queued ${#JOBS[@]} jobs"

launch_job () {
    local job="$1"
    local gpu="$2"
    read -r name method thresh mode ttbbr <<< "$job"
    local outdir="$SWEEP_ROOT/$name"

    # Idempotency: skip if already completed
    if [ -f "$outdir/stdout.log" ] && grep -q "AP50_ALL_mean:" "$outdir/stdout.log"; then
        echo "[sweep-xai] SKIP $name ($(grep AP50_ALL_mean $outdir/stdout.log | tail -1))"
        return
    fi
    # Skip if a window with this name already exists (job in flight)
    if tmux list-windows -t xai-short 2>/dev/null | grep -qE "[0-9]+: ${name}[- *]?"; then
        echo "[sweep-xai] SKIP $name (window already exists)"
        return
    fi

    mkdir -p "$outdir"
    echo "[sweep-xai] LAUNCH $name  gpu=$gpu  method=$method thresh=$thresh mode=$mode ttbbr=$ttbbr"

    local ttbbr_flag="False"
    [[ "$ttbbr" == "1" ]] && ttbbr_flag="True"

    local window="${name}"
    tmux new-window -t xai-short -n "$window" \
        "cd /home/ilias/AMROD && \
         HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose run --rm \
           -e CUDA_VISIBLE_DEVICES=$gpu \
           -e DETECTRON2_DATASETS=$DATASETS \
           -e PYTHONPATH=/workspace/AMROD/detectron2:/workspace/AMROD \
           amrod bash -lc 'cd detectron2/tools && python adapt.py \
             --config-file $CFG \
             MODEL.WEIGHTS $CKPT \
             OUTPUT_DIR $outdir \
             SOLVER.XAI_FILTER_ENABLED True \
             SOLVER.XAI_METHOD $method \
             SOLVER.XAI_THRESHOLD $thresh \
             SOLVER.XAI_MODE $mode \
             SOLVER.TTBBR_ENABLED $ttbbr_flag \
             SOLVER.TTBBR_IOU_THRESH 0.7 \
             SOLVER.TTBBR_DROP_INCONSISTENT False' \
         2>&1 | tee $outdir/stdout.log; echo '=== DONE: '$name' ==='; \
         sleep 3; \
         tmux kill-window -t xai-short:$window"
}

# Dispatch: sequential (NUM_GPUS=1). Wait until fewer than NUM_GPUS non-control
# non-dispatcher windows are alive before launching the next job. Excluding
# BOTH `control` and `dispatcher` is required, otherwise the dispatcher window
# self-counts and deadlocks the sweep.
count_active_jobs () {
    tmux list-windows -t xai-short 2>/dev/null \
        | grep -vE "^[0-9]+: (control|dispatcher)[- *]?" \
        | wc -l
}

for i in "${!JOBS[@]}"; do
    while [ "$(count_active_jobs)" -ge "$NUM_GPUS" ]; do
        sleep 10
    done
    gpu=${GPU_LIST[$(( i % NUM_GPUS ))]}
    launch_job "${JOBS[$i]}" "$gpu"
    sleep 3
done

echo "[sweep-xai] all jobs dispatched. Watch progress:"
echo "  tmux attach -t xai-short"
echo "Aggregate results:"
echo "  python3 /home/ilias/AMROD/tools/aggregate_xai_sweep.py"
