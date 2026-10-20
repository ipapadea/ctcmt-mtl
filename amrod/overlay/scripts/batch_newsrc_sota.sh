#!/usr/bin/env bash
set -Eeuo pipefail

# Safe new-source SOTA batch. Never deletes or truncates an experiment log.
# Usage:
#   DRY_RUN=1 bash scripts/batch_newsrc_sota.sh 0 1
#   CONFIRM_CSC12=1 bash scripts/batch_newsrc_sota.sh 0 1
#   ONLY=cscLT CONFIRM_CSC12=1 bash scripts/batch_newsrc_sota.sh 0 1
# Ctrl+C/TERM: stop both queues and their foreground runner process groups.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
CFG="detectron2/configs/Cityscapes"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
SOURCE_NAME="panoptic_fpn_R50_cityscapes_segw1"
SOURCE_WEIGHTS="/workspace/output/${SOURCE_NAME}/model_final.pth"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"
GPUS=("$@")
((${#GPUS[@]})) || GPUS=(0 1)

# CSC-12 list verified from the original completed AMROD CSC-12 log.
# Exact CSC-12 order from the original successful AMROD short-term benchmark.
CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
       brightness contrast elastic_transform pixelate jpeg_compression)
CSC5=(fog motion_blur snow brightness defocus_blur)
ACDC4=(fog night rain snow)

stream_for() {
    local proto="$1" name="$2" c r s="("; local -a names=()
    case "$proto" in
        csc12) names=("${CSC12[@]}") ;;
        cscLT) for ((r=0; r<10; r++)); do names+=("${CSC5[@]}"); done ;;
        acdc4) for c in "${ACDC4[@]}"; do names+=("acdc_${c}"); done ;;
        acdcLT) for ((r=0; r<10; r++)); do
                    for c in "${ACDC4[@]}"; do names+=("acdc_${c}"); done
                done ;;
        *) echo "Unknown protocol: $proto" >&2; return 2 ;;
    esac
    # CoTTA and TENT register segmentation-only datasets under *_semseg.
    # AMROD uses unsuffixed datasets. Runners override DATASETS.TEST with STREAM.
    if [[ "$name" == cotta_* || "$name" == tent_* ]]; then
        for i in "${!names[@]}"; do names[i]="${names[i]}_semseg"; done
    fi
    for c in "${names[@]}"; do s+="\"${c}\","; done
    printf '%s' "${s%,})"
}

# Internal worker, always launched as an independent session/process group.
# All foreground runner descendants inherit the same group, so the parent can
# stop both queues with kill -TERM -- -<worker-pgid>.
if [[ "${1:-}" == "--worker" ]]; then
    gpu="$2"; queue_file="$3"
    trap 'echo "[GPU ${gpu}] stopped" >&2; exit 143' INT TERM HUP
    while IFS=$'\t' read -r proto name config expected; do
        [[ -n "$proto" ]] || continue
        stream="$(stream_for "$proto" "$name")"
        case "$proto" in
            csc*) runner="${HERE}/run_mixed_lt_local.sh" ;;
            acdc*) runner="${HERE}/run_ctcr_acdc_local.sh" ;;
            *) echo "Bad protocol: $proto" >&2; exit 2 ;;
        esac
        echo "[GPU $gpu] START $name $(date -Is)"
        # Foreground runner: no daemonization and no detached subprocess here.
        if STREAM="$stream" bash "$runner" "$gpu" "$name" "$config" "$SEED"; then
            log="${HOST_OUT}/logs/${name}.log"
            actual=0; errors=0
            if [[ -f "$log" ]]; then
                actual="$(grep -c 'in csv format' "$log" || true)"
                errors="$(grep -Ec 'Traceback|CUDA out of memory|^RuntimeError:' "$log" || true)"
            fi
            if ((actual != expected || errors != 0)); then
                echo "[GPU $gpu] INCOMPLETE $name: ${actual}/${expected} evals, ${errors} errors" >&2
                exit 4
            fi
            echo "[GPU $gpu] DONE $name $(date -Is)"
        else
            rc=$?
            echo "[GPU $gpu] FAILED $name exit=$rc" >&2
            exit "$rc"
        fi
    done < "$queue_file"
    exit 0
fi

JOBS=(
    "cscLT:amrod_newsrc_cscLT_s${SEED}:${CFG}/amrod_newsrc_csc.yaml:50"
    "acdcLT:amrod_newsrc_acdcLT_s${SEED}:${CFG}/amrod_newsrc_acdc.yaml:40"
    "csc12:amrod_newsrc_csc12_s${SEED}:${CFG}/amrod_newsrc_csc.yaml:12"
    "acdc4:amrod_newsrc_acdc4_s${SEED}:${CFG}/amrod_newsrc_acdc.yaml:4"
    "cscLT:cotta_newsrc_cscLT_s${SEED}:${CFG}/cotta_newsrc_csc.yaml:50"
    "acdcLT:cotta_newsrc_acdcLT_s${SEED}:${CFG}/cotta_newsrc_acdc.yaml:40"
    "csc12:cotta_newsrc_csc12_s${SEED}:${CFG}/cotta_newsrc_csc.yaml:12"
    "acdc4:cotta_newsrc_acdc4_s${SEED}:${CFG}/cotta_newsrc_acdc.yaml:4"
    "cscLT:tent_newsrc_cscLT_s${SEED}:${CFG}/tent_newsrc_csc.yaml:50"
    "acdcLT:tent_newsrc_acdcLT_s${SEED}:${CFG}/tent_newsrc_acdc.yaml:40"
    "csc12:tent_newsrc_csc12_s${SEED}:${CFG}/tent_newsrc_csc.yaml:12"
    "acdc4:tent_newsrc_acdc4_s${SEED}:${CFG}/tent_newsrc_acdc.yaml:4"
)

# Strict preflight, including source weights for the newly copied SOTA YAMLs.
[[ -f "${HOST_OUT}/${SOURCE_NAME}/model_final.pth" ]] || {
    echo "ERROR: new-source checkpoint missing in ${HOST_OUT}/${SOURCE_NAME}" >&2; exit 2;
}
for runner in run_mixed_lt_local.sh run_ctcr_acdc_local.sh; do
    [[ -f "${HERE}/${runner}" ]] || { echo "ERROR: missing ${runner}" >&2; exit 2; }
done
for f in cotta_newsrc_csc.yaml cotta_newsrc_acdc.yaml \
         tent_newsrc_csc.yaml tent_newsrc_acdc.yaml; do
    path="${ROOT}/${CFG}/${f}"
    [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }
    grep -Fq "$SOURCE_WEIGHTS" "$path" || {
        echo "ERROR: $f does not directly set new-source MODEL.WEIGHTS" >&2; exit 2;
    }
done
for job in "${JOBS[@]}"; do
    IFS=: read -r proto name config expected <<< "$job"
    [[ -f "${ROOT}/${config}" ]] || { echo "ERROR: missing ${config}" >&2; exit 2; }
done

TODO=()
for job in "${JOBS[@]}"; do
    IFS=: read -r proto name config expected <<< "$job"
    if [[ "$ONLY" != all && "$ONLY" != "$proto" ]]; then continue; fi
    log="${HOST_OUT}/logs/${name}.log"
    count=0; errors=0
    if [[ -f "$log" ]]; then
        count="$(grep -c 'in csv format' "$log" || true)"
        errors="$(grep -Ec 'Traceback|CUDA out of memory|^RuntimeError:' "$log" || true)"
        if ((count == expected && errors == 0)); then
            printf 'SKIP %-42s (%s/%s)\n' "$name" "$count" "$expected"
            continue
        fi
        echo "ERROR: existing incomplete/failed log: ${name} (${count}/${expected}, errors=${errors})." >&2
        echo "Move it aside manually only after inspecting it; nothing was overwritten." >&2
        exit 3
    fi
    TODO+=("$job")
done
printf 'TODO: %s job(s), GPUs: %s, ONLY=%s\n' "${#TODO[@]}" "${GPUS[*]}" "$ONLY"
for i in "${!TODO[@]}"; do
    IFS=: read -r proto name config expected <<< "${TODO[i]}"
    printf '  GPU %s  %-7s %s (%s evals)\n' "${GPUS[i % ${#GPUS[@]}]}" "$proto" "$name" "$expected"
done

# This guard is BEFORE any job launches. Config/log reads above are read-only.
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry-run stream verification (first 4 entries, actual counts):"
    for spec in 'csc12 amrod_newsrc_csc12_s0' 'csc12 cotta_newsrc_csc12_s0' \
                'cscLT tent_newsrc_cscLT_s0' 'acdc4 amrod_newsrc_acdc4_s0' \
                'acdcLT cotta_newsrc_acdcLT_s0'; do
        read -r proto name <<< "$spec"
        stream="$(stream_for "$proto" "$name")"
        count="$(grep -o ',' <<< "$stream" | wc -l)"
        # Elements = comma-separated items, count = commas + 1.
        count=$((count+1))
        preview="$(printf '%s' "$stream" | cut -d, -f1-4)"
        printf '  %-7s %-29s %2d datasets: %s ...\n' "$proto" "$name" "$count" "$preview"
    done
    echo "DRY_RUN=1: no runs or worker processes launched."
    exit 0
fi
((${#TODO[@]})) || { echo 'Nothing to run.'; exit 0; }
if [[ "$ONLY" == all || "$ONLY" == csc12 ]]; then
    [[ "${CONFIRM_CSC12:-0}" == 1 ]] || {
        echo "ERROR: verify CSC12 names against the original CSC-12corr benchmark;" >&2
        echo "then set CONFIRM_CSC12=1 to authorize real short runs." >&2
        exit 2
    }
fi

# Put each complete GPU queue (including its runner children) in ONE process
# group. This avoids orphaning foreground runner processes when Ctrl+C is sent.
mkdir -p "${HOST_OUT}/logs"
queue_dir="$(mktemp -d "${HOST_OUT}/logs/.newsrc_sota_queue.XXXXXX")"
workers=()
cleanup_parent() {
    local p
    trap - INT TERM HUP
    echo 'Stopping both GPU process groups...' >&2
    for p in "${workers[@]}"; do
        kill -TERM -- "-${p}" 2>/dev/null || true
    done
    sleep 2
    for p in "${workers[@]}"; do
        if kill -0 -- "-${p}" 2>/dev/null; then
            kill -KILL -- "-${p}" 2>/dev/null || true
        fi
    done
    for p in "${workers[@]}"; do wait "$p" 2>/dev/null || true; done
    rm -rf -- "$queue_dir"
}
trap 'cleanup_parent; exit 130' INT TERM HUP
for g in "${!GPUS[@]}"; do : > "${queue_dir}/${g}.tsv"; done
for i in "${!TODO[@]}"; do
    IFS=: read -r proto name config expected <<< "${TODO[i]}"
    g=$((i % ${#GPUS[@]}))
    printf '%s\t%s\t%s\t%s\n' "$proto" "$name" "$config" "$expected" \
        >> "${queue_dir}/${g}.tsv"
done
for g in "${!GPUS[@]}"; do
    # New session PGID is the background setsid PID in ordinary non-job-control
    # bash. Check it once if the local setsid behaves differently.
    setsid bash "$0" --worker "${GPUS[g]}" "${queue_dir}/${g}.tsv" &
    workers+=("$!")
done
fail=0
for p in "${workers[@]}"; do
    if wait "$p"; then :; else fail=1; fi
done
trap - INT TERM HUP
rm -rf -- "$queue_dir"
((fail == 0)) || { echo 'One or more GPU queues failed; inspect logs.' >&2; exit 1; }
echo 'All selected jobs finished. Verify logs before gathering results.'
