#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# New-source SOTA comparison
#
# Methods:
#   AMROD
#   CoTTA
#   TENT
#
# Protocols:
#   CSC-short   = 12 corruptions x1
#   ACDC-short  = 4 domains x1
#   CSC-LT      = 5 corruptions x10 = 50 evals
#   ACDC-LT     = 4 domains x10 = 40 evals
#
# Usage:
#   bash scripts/batch_newsrc_sota.sh 0 1
#
# GPU0 and GPU1 are used as two sequential queues.
# Completed runs are skipped.
# ============================================================

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

export HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
export HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
export CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
export CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
export ACDC_ROOT="${ACDC_ROOT:-/media/ilias/DATA/ilias/acdc}"

CFG="detectron2/configs/Cityscapes"
SEED="${SEED:-0}"

GPUS=("$@")
[[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0 1)

# ------------------------------------------------------------
# Streams
# ------------------------------------------------------------

# CSC short:
# 12 corruption datasets, one pass each.
CSC12=(
    fog
    frost
    snow
    brightness
    contrast
    defocus_blur
    glass_blur
    motion_blur
    zoom_blur
    gaussian_noise
    impulse_noise
    shot_noise
)

CSC_SHORT_STREAM="("
for c in "${CSC12[@]}"; do
    CSC_SHORT_STREAM+="\"${c}\","
done
CSC_SHORT_STREAM="${CSC_SHORT_STREAM%,})"

# CSC long:
# exact protocol already used in our main experiments.
CSC_CYCLE=(fog motion_blur snow brightness defocus_blur)

CSC_LT_STREAM="("
for _ in $(seq 1 10); do
    for c in "${CSC_CYCLE[@]}"; do
        CSC_LT_STREAM+="\"${c}\","
    done
done
CSC_LT_STREAM="${CSC_LT_STREAM%,})"

# ACDC short.
ACDC_CYCLE=(fog night rain snow)

ACDC_SHORT_STREAM="("
for c in "${ACDC_CYCLE[@]}"; do
    ACDC_SHORT_STREAM+="\"acdc_${c}\","
done
ACDC_SHORT_STREAM="${ACDC_SHORT_STREAM%,})"

# ACDC long.
ACDC_LT_STREAM="("
for _ in $(seq 1 10); do
    for c in "${ACDC_CYCLE[@]}"; do
        ACDC_LT_STREAM+="\"acdc_${c}\","
    done
done
ACDC_LT_STREAM="${ACDC_LT_STREAM%,})"


# ------------------------------------------------------------
# Jobs
#
# proto:name:config:expected_evals
# ------------------------------------------------------------

JOBS=(

    # ========================================================
    # AMROD
    # ========================================================

    # LT runs already completed -> automatically skipped.
    "cscLT:amrod_newsrc_cscLT_s${SEED}:${CFG}/amrod_newsrc_csc.yaml:50"
    "acdcLT:amrod_newsrc_acdcLT_s${SEED}:${CFG}/amrod_newsrc_acdc.yaml:40"

    # Short protocols
    "csc12:amrod_newsrc_csc12_s${SEED}:${CFG}/amrod_newsrc_csc.yaml:12"
    "acdc4:amrod_newsrc_acdc4_s${SEED}:${CFG}/amrod_newsrc_acdc.yaml:4"


    # ========================================================
    # CoTTA
    # ========================================================

    "cscLT:cotta_newsrc_cscLT_s${SEED}:${CFG}/cotta_newsrc_csc.yaml:50"
    "acdcLT:cotta_newsrc_acdcLT_s${SEED}:${CFG}/cotta_newsrc_acdc.yaml:40"

    "csc12:cotta_newsrc_csc12_s${SEED}:${CFG}/cotta_newsrc_csc.yaml:12"
    "acdc4:cotta_newsrc_acdc4_s${SEED}:${CFG}/cotta_newsrc_acdc.yaml:4"


    # ========================================================
    # TENT
    # ========================================================

    "cscLT:tent_newsrc_cscLT_s${SEED}:${CFG}/tent_newsrc_csc.yaml:50"
    "acdcLT:tent_newsrc_acdcLT_s${SEED}:${CFG}/tent_newsrc_acdc.yaml:40"

    "csc12:tent_newsrc_csc12_s${SEED}:${CFG}/tent_newsrc_csc.yaml:12"
    "acdc4:tent_newsrc_acdc4_s${SEED}:${CFG}/tent_newsrc_acdc.yaml:4"
)


# ------------------------------------------------------------
# Config sanity check
# ------------------------------------------------------------

echo
echo "Checking configs..."

missing=0

for job in "${JOBS[@]}"; do
    IFS=: read -r proto name cfg want <<< "${job}"

    if [[ ! -f "${ROOT}/${cfg}" ]]; then
        echo "MISSING: ${cfg}"
        missing=1
    fi
done

if [[ "${missing}" -ne 0 ]]; then
    echo
    echo "One or more configs are missing."
    echo "Create/fix them before launching the batch."
    exit 2
fi


# ------------------------------------------------------------
# Determine TODO jobs
# ------------------------------------------------------------

TODO=()

echo
echo "Checking existing results..."

for job in "${JOBS[@]}"; do

    IFS=: read -r proto name cfg want <<< "${job}"

    log="${HOST_OUT}/logs/${name}.log"
    n=0

    if [[ -f "${log}" ]]; then
        n="$(grep -c 'in csv format' "${log}" || true)"
    fi

    if [[ "${n}" -eq "${want}" ]]; then

        echo "SKIP  ${name}  (${n}/${want})"

    else

        echo "TODO  ${name}  (${n}/${want})"
        TODO+=("${job}")

    fi

done

if [[ ${#TODO[@]} -eq 0 ]]; then
    echo
    echo "Nothing to do."
    exit 0
fi


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

run_one () {

    gpu="$1"
    job="$2"

    IFS=: read -r proto name cfg want <<< "${job}"

    echo
    echo "================================================================"
    echo "[GPU ${gpu}] START ${name}"
    echo "Protocol: ${proto}"
    echo "Config  : ${cfg}"
    echo "Time    : $(date)"
    echo "================================================================"

    case "${proto}" in

        cscLT)

            STREAM="${CSC_LT_STREAM}" \
            bash "${HERE}/run_mixed_lt_local.sh" \
                "${gpu}" \
                "${name}" \
                "${cfg}" \
                "${SEED}"
            ;;

        csc12)

            STREAM="${CSC_SHORT_STREAM}" \
            bash "${HERE}/run_mixed_lt_local.sh" \
                "${gpu}" \
                "${name}" \
                "${cfg}" \
                "${SEED}"
            ;;

        acdcLT)

            STREAM="${ACDC_LT_STREAM}" \
            bash "${HERE}/run_ctcr_acdc_local.sh" \
                "${gpu}" \
                "${name}" \
                "${cfg}" \
                "${SEED}"
            ;;

        acdc4)

            STREAM="${ACDC_SHORT_STREAM}" \
            bash "${HERE}/run_ctcr_acdc_local.sh" \
                "${gpu}" \
                "${name}" \
                "${cfg}" \
                "${SEED}"
            ;;

        *)

            echo "Unknown protocol: ${proto}"
            return 2
            ;;

    esac

    echo
    echo "[GPU ${gpu}] DONE ${name}"
    echo "Time: $(date)"
}


# ------------------------------------------------------------
# Parallel GPU queues
# ------------------------------------------------------------

echo
echo "================================================================"
echo "NEW-SOURCE SOTA BATCH"
echo "Seed : ${SEED}"
echo "Jobs : ${#TODO[@]}"
echo "GPUs : ${GPUS[*]}"
echo "================================================================"

for i in "${!TODO[@]}"; do
    IFS=: read -r proto name cfg want <<< "${TODO[$i]}"

    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"

    echo "GPU ${gpu}  ${proto}  ${name}"
done

echo "================================================================"

pids=()

for g in "${!GPUS[@]}"; do

    (
        for i in "${!TODO[@]}"; do

            [[ $((i % ${#GPUS[@]})) -eq "${g}" ]] || continue

            run_one \
                "${GPUS[$g]}" \
                "${TODO[$i]}" \
            || echo "!!! FAILED ${TODO[$i]}"

        done
    ) &

    pids+=($!)

done


for p in "${pids[@]}"; do
    wait "${p}" || true
done


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

echo
echo "================================================================"
echo "BATCH FINISHED"
echo "================================================================"

for job in "${JOBS[@]}"; do

    IFS=: read -r proto name cfg want <<< "${job}"

    log="${HOST_OUT}/logs/${name}.log"

    if [[ -f "${log}" ]]; then

        n="$(grep -c 'in csv format' "${log}" || true)"
        tr="$(grep -c 'Traceback' "${log}" || true)"

        printf "%-42s %3d/%-3d evals  traceback=%s\n" \
            "${name}" "${n}" "${want}" "${tr}"

    else

        printf "%-42s NO LOG\n" "${name}"

    fi

done
