#!/usr/bin/env bash
# Long-term CTTA (paper Table 3 protocol) + TT-BBR + 5-seed variance report.
#
# 5 corruptions x 10 rounds continuous adaptation on GPU 3 (cuda device 2).
# Total per run: 25000 iterations, ~1.5 hours wall time.
#
# Seed pattern mirrors the short-term GPU 4 sweep:
#   1 unseeded (SEED=-1 -> random), plus 4 fixed seeds (SEED = 1, 2, 3, 4).
# Total: 5 runs, ~7-8 hours sequential wall time on GPU 3.
#
# Usage:
#   ./scripts/sweep_ttbbr_long.sh
#
# Idempotent: skips runs whose stdout.log already contains AP50_ALL_mean.
set -euo pipefail
# ---- Auto-relaunch inside tmux so the dispatcher survives shell close ----
# Detects two failure modes we hit earlier:
#   (1) User runs `bash sweep_ttbbr_long.sh` in a shell => dispatcher loop
#       dies if they Ctrl-C or close the terminal, leaving mid-sweep jobs
#       unqueued.
#   (2) User runs inside another tmux session (`ttbbr-sweep`) by mistake;
#       we'd launch our windows in the wrong session.
# Re-exec inside a dedicated `ttbbr-long` tmux session and return control
# to the caller immediately. Any child of the dispatcher window survives
# shell close.
if [ "${TTBBR_LONG_IN_TMUX:-0}" != "1" ]; then
    export TTBBR_LONG_IN_TMUX=1
    tmux has-session -t ttbbr-long 2>/dev/null || \
        tmux new-session -d -s ttbbr-long -n control
    tmux new-window -t ttbbr-long -n dispatcher \
        "TTBBR_LONG_IN_TMUX=1 bash '$0' $* 2>&1 | tee /tmp/ttbbr_long_dispatcher.log; exec bash -l"
    echo "[sweep-long] dispatcher launched inside tmux session 'ttbbr-long'."
    echo "[sweep-long] Attach:   tmux attach -t ttbbr-long"
    echo "[sweep-long] Aggregate: python3 /home/ilias/AMROD/tools/aggregate_ttbbr_long.py"
    echo "[sweep-long] Terminal is now free to close; runs continue in tmux."
    exit 0
fi
# GPU 3 in human-1-indexed terms = cuda:2.
GPU=2

SWEEP_ROOT=/data/vgcmt/work_dirs/ttbbr_long
mkdir -p "$SWEEP_ROOT"

CKPT=/data/vgcmt/models/amrod_d2/cityscapes_train_final.pth
DATASETS=/data/vgcmt/datasets/cityscapes_c_amrod
CFG=cfg_cityscapes_c_long_ttbbr.yaml
IOU=0.7
DROP=False

# ---- 5 seeds: 1 unseeded (random) + 4 fixed --------------------------
declare -a JOBS=(
    "seedRAND -1"
    "seed1     1"
    "seed2     2"
    "seed3     3"
    "seed4     4"
)

echo "[sweep-long] ${#JOBS[@]} runs queued for GPU $GPU"

# Create tmux session
tmux has-session -t ttbbr-long 2>/dev/null || tmux new-session -d -s ttbbr-long -n control

launch_job () {
    local name="$1"; local seed="$2"
    local outdir="$SWEEP_ROOT/$name"

    # Idempotency: skip if already completed
    if [ -f "$outdir/stdout.log" ] && grep -q "AP50_ALL_mean:" "$outdir/stdout.log"; then
        echo "[sweep-long] SKIP $name (done: $(grep AP50_ALL_mean $outdir/stdout.log | tail -1))"
        return
    fi
    if tmux list-windows -t ttbbr-long 2>/dev/null | grep -qE "[0-9]+: ${name}[- *]?"; then
        echo "[sweep-long] SKIP $name (window already exists)"
        return
    fi

    mkdir -p "$outdir"
    echo "[sweep-long] LAUNCH $name  gpu=$GPU  seed=$seed"

    tmux new-window -t ttbbr-long -n "$name" \
        "cd /home/ilias/AMROD && \
         HOST_UID=\$(id -u) HOST_GID=\$(id -g) docker compose run --rm \
           -e CUDA_VISIBLE_DEVICES=$GPU \
           -e DETECTRON2_DATASETS=$DATASETS \
           -e PYTHONPATH=/workspace/AMROD/detectron2:/workspace/AMROD \
           amrod bash -lc 'cd detectron2/tools && python adapt.py \
             --config-file $CFG \
             MODEL.WEIGHTS $CKPT \
             OUTPUT_DIR $outdir \
             SOLVER.TTBBR_IOU_THRESH $IOU \
             SOLVER.TTBBR_DROP_INCONSISTENT $DROP \
             SEED $seed' \
         2>&1 | tee $outdir/stdout.log; echo '=== DONE: '$name' ==='; \
         sleep 3; \
         tmux kill-window -t ttbbr-long:$name"
}

# Sequential dispatch (only 1 GPU, one job at a time).
# Exclude both the `control` window and the running `dispatcher` window itself
# from the active-job count -- otherwise self-counting causes a deadlock.
active_count () {
    tmux list-windows -t ttbbr-long 2>/dev/null \
        | grep -vE "^[0-9]+: (control|dispatcher)[- *]?" \
        | wc -l
}

for job in "${JOBS[@]}"; do
    read -r name seed <<< "$job"
    while [ "$(active_count)" -ge 1 ]; do
        sleep 15
    done
    launch_job "$name" "$seed"
    sleep 3
done

echo "[sweep-long] all jobs dispatched. Watch with:"
echo "  tmux attach -t ttbbr-long"
echo "Aggregate results:"
echo "  python3 /home/ilias/AMROD/tools/aggregate_ttbbr_long.py"
