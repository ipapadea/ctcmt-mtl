#!/usr/bin/env bash
# TT-BBR hyperparameter sweep with multi-seed variance measurement.
# Configured for GPU 3 only, sequential dispatch (one job at a time).
#
# Structure:
#   phase1 (7 runs): IoU threshold sweep, drop=False, seed=0
#       iou in {0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90}
#   phase2 (2 runs): drop-inconsistent ablation at iou=0.7 and 0.75, seed=0
#   phase3 (4 runs): multi-seed variance at BEST config (default iou=0.7, drop=False)
#       seed in {1, 2, 3, 4}  -- combined with the ORIGINAL un-seeded run at
#                                /data/vgcmt/work_dirs/amrod_ttbbr (mean 21.8),
#                                that gives 5 total seeds for the paper's
#                                variance report.
#
# Total NEW runs: 13. Sequential on GPU 3, ~30 min each = ~6.5 hours wall time.
#
# Usage:
#   ./scripts/sweep_ttbbr.sh phase1                      # 7 IoU sweep runs
#   ./scripts/sweep_ttbbr.sh phase2                      # 2 drop-inconsistent runs
#   ./scripts/sweep_ttbbr.sh phase3 [best_iou] [drop]    # 4 seed runs
#   ./scripts/sweep_ttbbr.sh all                         # phase1 + phase2 + phase3
#
# Idempotent: skips runs whose log.txt already contains AP50_ALL_mean.
set -euo pipefail

PHASE="${1:-all}"
BEST_IOU="${2:-0.70}"
BEST_DROP="${3:-False}"

SWEEP_ROOT=/data/vgcmt/work_dirs/ttbbr_sweep
mkdir -p "$SWEEP_ROOT"

# Single-GPU sequential dispatch on GPU 3.
GPU_LIST=(3)
NUM_GPUS=${#GPU_LIST[@]}

CKPT=/data/vgcmt/models/amrod_d2/cityscapes_train_final.pth
DATASETS=/data/vgcmt/datasets/cityscapes_c_amrod
CFG=cfg_cityscapes_c_short_ttbbr.yaml

# Job queue: each entry is "name iou drop seed"
declare -a JOBS=()

# ---- Populate phase1: IoU sweep -------------------------------------
if [[ "$PHASE" == "phase1" || "$PHASE" == "all" ]]; then
    for iou in 0.50 0.60 0.70 0.75 0.80 0.85 0.90; do
        name="iou${iou}_drop0_seed0"
        JOBS+=("$name $iou False 0")
    done
fi

# ---- Populate phase2: drop-inconsistent ablation ---------------------
if [[ "$PHASE" == "phase2" || "$PHASE" == "all" ]]; then
    for iou in 0.70 0.75; do
        name="iou${iou}_drop1_seed0"
        JOBS+=("$name $iou True 0")
    done
fi

# ---- Populate phase3: multi-seed at best config -----------------------
# 4 new seeds; combined with the ORIGINAL amrod_ttbbr result (random seed
# 2501266, mean 21.8), gives 5 total for the paper's variance estimate.
if [[ "$PHASE" == "phase3" || "$PHASE" == "all" ]]; then
    drop_flag=$( [[ "$BEST_DROP" == "True" || "$BEST_DROP" == "1" ]] && echo "1" || echo "0" )
    for seed in 1 2 3 4; do
        name="iou${BEST_IOU}_drop${drop_flag}_seed${seed}"
        JOBS+=("$name $BEST_IOU $BEST_DROP $seed")
    done
fi

echo "[sweep] queued ${#JOBS[@]} jobs"

# ---- Dispatch loop with GPU rotation and parallelism cap -------------
launch_job () {
    local job="$1"
    local gpu="$2"
    read -r name iou drop seed <<< "$job"
    local outdir="$SWEEP_ROOT/$name"

    # Idempotency: skip if already completed
    if [ -f "$outdir/stdout.log" ] && grep -q "AP50_ALL_mean:" "$outdir/stdout.log"; then
        echo "[sweep] SKIP $name (already done: $(grep AP50_ALL_mean $outdir/stdout.log | tail -1))"
        return
    fi
    # Skip if a window with this name already exists (job in flight from a prior invocation)
    if tmux list-windows -t ttbbr-sweep 2>/dev/null | grep -qE "[0-9]+: ${name}[- *]?"; then
        echo "[sweep] SKIP $name (window already exists in ttbbr-sweep session)"
        return
    fi

    mkdir -p "$outdir"
    echo "[sweep] LAUNCH $name  gpu=$gpu  iou=$iou drop=$drop seed=$seed"

    # Launch in a dedicated tmux window under the ttbbr-sweep session.
    # Tee to stdout.log (D2's own logger writes to log.txt in the same dir;
    # AP50_ALL_mean only appears on stdout so we capture it separately).
    local window="${name}"
    tmux new-window -t ttbbr-sweep -n "$window" \
        "cd /home/ilias/AMROD && \
         HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose run --rm \
           -e CUDA_VISIBLE_DEVICES=$gpu \
           -e DETECTRON2_DATASETS=$DATASETS \
           -e PYTHONPATH=/workspace/AMROD/detectron2:/workspace/AMROD \
           amrod bash -lc 'cd detectron2/tools && python adapt.py \
             --config-file $CFG \
             MODEL.WEIGHTS $CKPT \
             OUTPUT_DIR $outdir \
             SOLVER.TTBBR_IOU_THRESH $iou \
             SOLVER.TTBBR_DROP_INCONSISTENT $drop \
             SEED $seed' \
         2>&1 | tee $outdir/stdout.log; echo '=== DONE: '$name' ==='; \
         sleep 3; \
         tmux kill-window -t ttbbr-sweep:$window"
}

# Create the tmux session if missing
tmux has-session -t ttbbr-sweep 2>/dev/null || tmux new-session -d -s ttbbr-sweep -n control

# Dispatch: 4 concurrent jobs (one per GPU); wait for a slot before launching next.
# Exclude both the `control` window and the running `dispatcher` window itself
# from the active-job count -- otherwise self-counting causes a deadlock.
count_active_jobs () {
    tmux list-windows -t ttbbr-sweep 2>/dev/null \
        | grep -vE "^[0-9]+: (control|dispatcher)[- *]?" \
        | wc -l
}

for i in "${!JOBS[@]}"; do
    # Wait for a free GPU (window count < NUM_GPUS)
    while [ "$(count_active_jobs)" -ge "$NUM_GPUS" ]; do
        sleep 10
    done
    gpu=${GPU_LIST[$(( i % NUM_GPUS ))]}
    launch_job "${JOBS[$i]}" "$gpu"
    sleep 3   # small stagger so docker compose commands don't race
done

echo "[sweep] all jobs dispatched. Watch progress with:"
echo "  tmux attach -t ttbbr-sweep"
echo "Aggregate results with:"
echo "  python3 /home/ilias/AMROD/tools/aggregate_ttbbr_sweep.py"
