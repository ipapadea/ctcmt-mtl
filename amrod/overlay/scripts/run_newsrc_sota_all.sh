#!/usr/bin/env bash
set -Eeuo pipefail

# Automated new-source smoke checks, with optional hand-off to the full SOTA batch.
# Place next to batch_newsrc_sota.sh in AMROD/scripts/.
#   DRY_RUN=1 bash scripts/run_newsrc_sota_all.sh 0 1
#   bash scripts/run_newsrc_sota_all.sh 0 1             # all 4 smoke tests
#   RUN_FULL=1 bash scripts/run_newsrc_sota_all.sh 0 1  # smokes, then full batch
# Does not delete or replace prior experiment logs or output directories.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
LOGDIR="${HOST_OUT}/logs"
CFG="detectron2/configs/Cityscapes"
WEIGHTS="/workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
SEED="${SEED:-0}"
CSC_GPU="${1:-0}"
ACDC_GPU="${2:-1}"
RUN_FULL="${RUN_FULL:-0}"

cd "$ROOT"
[[ -f "${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || {
    echo "ERROR: new source checkpoint not found under HOST_OUT=$HOST_OUT" >&2
    exit 2
}
[[ -f "${HERE}/batch_newsrc_sota.sh" ]] || {
    echo 'ERROR: scripts/batch_newsrc_sota.sh not found.' >&2
    exit 2
}
for method in cotta tent; do
    for domain in csc acdc; do
        config="${CFG}/${method}_newsrc_${domain}.yaml"
        [[ -f "$config" ]] || { echo "ERROR: Missing $config" >&2; exit 2; }
        grep -Fq "$WEIGHTS" "$config" || {
            echo "ERROR: $config does not explicitly reference the segw1 checkpoint." >&2
            exit 2
        }
    done
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo 'SMOKE DRY RUN: would run CoTTA CSC, CoTTA ACDC, TENT CSC, TENT ACDC.'
    echo 'Then checking the full batch without launching anything:'
    DRY_RUN=1 bash "${HERE}/batch_newsrc_sota.sh" "$CSC_GPU" "$ACDC_GPU"
    echo 'DRY_RUN completed: no experiment launched.'
    exit 0
fi

mkdir -p "$LOGDIR/smoke_launch"
# Unique name so interrupted/old smoke logs are never overwritten.
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

run_smoke() {
    local method="$1" domain="$2" gpu="$3"
    local name="smoke_${method}_newsrc_${domain}_s${SEED}_${STAMP}"
    local config="${CFG}/${method}_newsrc_${domain}.yaml"
    local runner stream
    if [[ "$domain" == csc ]]; then
        runner="${HERE}/run_mixed_lt_local.sh"
        stream='("fog_semseg",)'
    else
        runner="${HERE}/run_ctcr_acdc_local.sh"
        stream='("acdc_fog_semseg",)'
    fi
    local launch_log="${LOGDIR}/smoke_launch/${name}.launch.log"
    local exp_log="${LOGDIR}/${name}.log"
    [[ ! -e "$launch_log" && ! -e "$exp_log" ]] || {
        echo "ERROR: Refusing to overwrite existing smoke log for $name" >&2
        return 2
    }
    echo "[GPU $gpu] START $name — STREAM=$stream"
    if STREAM="$stream" bash "$runner" "$gpu" "$name" "$config" "$SEED" \
        >"$launch_log" 2>&1; then
        :
    else
        local rc=$?
        echo "FAILED $name (exit $rc). Last output:" >&2
        tail -35 "$launch_log" >&2
        return "$rc"
    fi
    if [[ ! -f "$exp_log" ]]; then
        echo "FAILED $name: experiment log not found; launch log: $launch_log" >&2
        tail -35 "$launch_log" >&2
        return 3
    fi
    local evals errors
    evals="$(grep -c 'in csv format' "$exp_log" || true)"
    errors="$(grep -Eic 'Traceback|CUDA out of memory|RuntimeError:|ValueError:' "$exp_log" || true)"
    if [[ "$evals" != 1 || "$errors" != 0 ]] || \
       ! grep -q 'Task: sem_seg' "$exp_log"; then
        echo "FAILED $name: evals=$evals (expected 1), errors=$errors, sem_seg=$(grep -c 'Task: sem_seg' "$exp_log" || true)" >&2
        echo "Experiment log: $exp_log" >&2
        tail -35 "$exp_log" >&2
        return 4
    fi
    echo "PASS $name: 1 evaluation, sem_seg present, no detected error"
}

# Sequential smoke checks intentionally avoid leaving parallel jobs behind on failure.
for method in cotta tent; do
    run_smoke "$method" csc "$CSC_GPU"
    run_smoke "$method" acdc "$ACDC_GPU"
done

echo 'ALL 4 SMOKE TESTS PASSED.'
if [[ "$RUN_FULL" == "1" ]]; then
    echo 'Running final read-only batch preflight...'
    DRY_RUN=1 bash "${HERE}/batch_newsrc_sota.sh" "$CSC_GPU" "$ACDC_GPU"
    echo 'Launching full SOTA batch on both GPUs...'
    # exec passes control/signals directly to the existing batch script.
    exec env CONFIRM_CSC12=1 bash "${HERE}/batch_newsrc_sota.sh" "$CSC_GPU" "$ACDC_GPU"
fi

echo 'The full batch has NOT started. To launch it after inspecting smoke logs:'
echo "  CONFIRM_CSC12=1 bash scripts/batch_newsrc_sota.sh $CSC_GPU $ACDC_GPU"
