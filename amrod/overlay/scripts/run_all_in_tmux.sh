#!/usr/bin/env bash
# One-shot: fires everything you need in SEPARATE tmux sessions and returns
# immediately.
#
#   ./scripts/run_all_in_tmux.sh
#
# After firing, attach with:
#   tmux attach -t amrod-eval-amrod    # GPU 2, D2 pipeline, AMROD ckpt
#   tmux attach -t amrod-eval-ours     # GPU 3, D2 pipeline, ours ckpt
#   tmux attach -t amrod-aggregate     # waits for both, prints cross-tab
#   tmux attach -t amrod-monitor       # nvidia-smi + status markers watch
# Detach any of them with:  Ctrl-b d
# List all sessions:         tmux ls
#
# Idempotent: eval_d2_cityscapes_c.sh already skips corruptions whose
# results.json is cached, so re-running is safe.
set -euo pipefail

STATUS=/data/vgcmt/status
mkdir -p "$STATUS"

# ---- Ckpt paths (INSIDE the amrod container) ----------------------------
AMROD_D2_CKPT="/data/vgcmt/models/amrod_d2/cityscapes_train_final.pth"
OURS_D2_CKPT="/data/vgcmt/work_dirs/ours_d2/ours_cityscapes_d2.pth"

# ---- Pre-flight: replace the AMROD D2 symlink with a real file if needed
# docker-compose.yml mounts /data/vgcmt/downloads:ro so the symlink resolves
# inside the container. If the symlink is dead (source deleted), materialize
# a real copy as belt-and-suspenders.
if [ ! -e "$AMROD_D2_CKPT" ]; then
    src="/data/vgcmt/downloads/amrod/extracted/model weight/cityscapes_train_final.pth"
    if [ -f "$src" ]; then
        echo "[preflight] symlink dead; copying real file..."
        rm -f "$AMROD_D2_CKPT"
        cp "$src" "$AMROD_D2_CKPT"
    else
        echo "[preflight] FATAL: AMROD ckpt not found at $src" >&2
        exit 1
    fi
fi
[ -f "$OURS_D2_CKPT" ] || { echo "[preflight] FATAL: ours-D2 ckpt missing at $OURS_D2_CKPT" >&2; exit 1; }
ls -la "$AMROD_D2_CKPT" "$OURS_D2_CKPT"

# ---- Helper: create-or-reuse a detached tmux session and send a command
fire_session () {
    local name="$1"; shift
    local cmd="$*"
    if tmux has-session -t "$name" 2>/dev/null; then
        # Reuse existing session: send fresh command (any previous work is
        # idempotent-cached at the per-corruption results.json level).
        tmux send-keys -t "$name" "$cmd" C-m
    else
        # New detached session; keep a shell alive after the command exits so
        # you can inspect output.
        tmux new-session -d -s "$name" \
            "bash -lc $(printf '%q' "$cmd"); exec bash -l"
    fi
}

# ---- Fire the D2 evaluations in parallel on GPUs 2 and 3 --------------
fire_session amrod-eval-amrod "\
cd /home/ilias/AMROD && CUDA_VISIBLE_DEVICES=2 \
  ./scripts/eval_d2_cityscapes_c.sh '${AMROD_D2_CKPT}' amrod_ckpt 2 \
  2>&1 | tee /tmp/eval_d2_amrod.log && \
touch ${STATUS}/d2_amrod.done && echo '[eval-amrod] DONE'"

fire_session amrod-eval-ours "\
cd /home/ilias/AMROD && CUDA_VISIBLE_DEVICES=3 \
  ./scripts/eval_d2_cityscapes_c.sh '${OURS_D2_CKPT}' ours_ckpt 3 \
  2>&1 | tee /tmp/eval_d2_ours.log && \
touch ${STATUS}/d2_ours.done && echo '[eval-ours] DONE'"

# ---- Aggregator waits for both, then prints the 2x2 cross-tab ---------
fire_session amrod-aggregate "\
until [ -f ${STATUS}/d2_amrod.done ] && [ -f ${STATUS}/d2_ours.done ]; do \
    echo \"[waiter] \$(date +%H:%M:%S) waiting for D2 evals (d2_amrod.done, d2_ours.done)...\"; \
    sleep 30; \
done && \
python3 /home/ilias/AMROD/tools/cross_tabulate.py 2>&1 | tee /tmp/cross_tab_final.txt && \
echo '' && echo '=== PIPELINE COMPLETE. Results at /tmp/cross_tab_final.txt ===' && \
touch ${STATUS}/all.done"

# ---- Monitor ----------------------------------------------------------
fire_session amrod-monitor "\
watch -n 5 'echo === GPUs ===; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; echo === Status markers ===; ls -1 ${STATUS}/ 2>/dev/null; echo; echo === D2 per-corruption results ===; find /data/vgcmt/work_dirs/amrod_d2_pipeline -name results.json 2>/dev/null | sort'"

echo
echo "===================================================================="
echo "Four independent tmux sessions launched:"
echo
echo "  tmux attach -t amrod-eval-amrod   # GPU 2, D2 + AMROD ckpt"
echo "  tmux attach -t amrod-eval-ours    # GPU 3, D2 + ours ckpt"
echo "  tmux attach -t amrod-aggregate    # waits + prints cross-tab"
echo "  tmux attach -t amrod-monitor      # nvidia-smi + status watch"
echo
echo "Detach any of them with:  Ctrl-b d"
echo "List all sessions:        tmux ls"
echo
echo "Marker files under ${STATUS}/:"
echo "  d2_amrod.done    d2_ours.done    all.done"
echo
echo "Final table:  /tmp/cross_tab_final.txt"
echo "Wall time estimate: ~6-8 min (12 corr x ~30s per GPU, parallel)."
echo "===================================================================="
