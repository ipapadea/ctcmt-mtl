#!/usr/bin/env bash
# Long-term CTTA batch: 10-round fog->night->rain->snow evaluation.
#
# Four parallel tmux sessions:
#   GPU 0  amrod           detectron2  Mask R-CNN
#   GPU 1  ctcmt_det       detectron2  det-only (single-task det baseline)
#   GPU 2  ctcmt_v4b       detectron2  Panoptic FPN MTL (best ACDC config)
#   GPU 3  ctcmt_trilitenet TriLiteNet MTL (our new backbone experiment)
#              *** GPU 3 requires training to be complete first ***
#              *** Use ckpt_best_det.pth once epoch 300 is done  ***
#
# Usage:
#   bash scripts/run_longtask_batch.sh           # launch all
#   bash scripts/run_longtask_batch.sh d2        # detectron2 runs only
#   bash scripts/run_longtask_batch.sh trilit     # TriLiteNet only
#
# Logs land in /tmp/longtask_*.log and workspace output dirs.
# Monitor with:  tmux ls  /  tmux attach -t longtask-<name>

set -euo pipefail

MODE="${1:-all}"   # all | d2 | trilit

LOGS=/tmp
RUNNER="bash /home/ilias/AMROD/scripts/run_ctta_acdc.sh"
N=10   # number of CTTA rounds

TRILIT_CKPT="runs/cityscapes_mtl_base_native/ckpt_best_det.pth"

_launch() {
    local name="$1"; shift
    tmux kill-session -t "longtask-${name}" 2>/dev/null || true
    tmux new-session -d -s "longtask-${name}" \
        "echo '[longtask-${name}] START' && $* 2>&1 | tee ${LOGS}/longtask_${name}.log; echo '[longtask-${name}] DONE'"
    echo "  launched longtask-${name}"
}

echo "=== Long-term CTTA batch (${N} rounds) ==="

if [[ "$MODE" == "all" || "$MODE" == "d2" ]]; then
    # GPU 2: all three d2 runs sequentially in one tmux session
    _launch "d2_seq" \
        "$RUNNER 2 amrod $N && \
         $RUNNER 2 ctcmt_det $N && \
         $RUNNER 2 ctcmt_v4b $N"
fi

if [[ "$MODE" == "all" || "$MODE" == "trilit" ]]; then
    # GPU 3: CT-CMT TriLiteNet (requires epoch-300 checkpoint)
    if [[ ! -f "/home/ilias/TriLiteNet/${TRILIT_CKPT}" ]]; then
        echo "  [WARN] TriLiteNet ckpt not found: ${TRILIT_CKPT}"
        echo "  [WARN] Skipping GPU3 TriLiteNet run — start it manually once training finishes."
    else
        _launch "ctcmt_trilit" \
            "docker run --rm --gpus '\"device=3\"' --shm-size=8g \
              -v /home/ilias/TriLiteNet:/workspace \
              -v /data/ilias/acdc:/data/ilias/acdc:ro \
              trilitenet:latest \
              python tools/adapt_ctta.py \
                --ckpt ${TRILIT_CKPT} \
                --method ctcmt \
                --num-seg-classes 19 --num-det-classes 8 \
                --num-repeats ${N} --device 0"
    fi
fi

echo ""
echo "=== Sessions launched ==="
tmux ls | grep longtask || true
echo ""
echo "Attach:  tmux attach -t longtask-<name>"
echo "Detach:  Ctrl-b d"
echo "Logs:    tail -f ${LOGS}/longtask_<name>.log"
echo ""
echo "When all done, collect results:"
echo "  grep -A2 'per-weather\\|CTTA finished' ${LOGS}/longtask_*.log"
