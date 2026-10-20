#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the remaining CTAOD/OBAO seed-0 experiments on two GPUs.
# The caller may disconnect after launching this script as a detached process.

ROOT="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CFG_ROOT="detectron2/configs/Cityscapes"
SEED=0
DRY_RUN="${DRY_RUN:-0}"

CHECKPOINT="${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
EXPECTED_MD5="5f63c7578cb925552fd1b9d69980937a"
META_ROOT="${HOST_OUT}/phase2_batch_meta"
LOG_ROOT="${HOST_OUT}/logs"
PATCH_ROOT="${HOST_OUT}/phase2_provenance"

CTAOD_EXP="newsrc_ctaod_cscLT_s0_rerun"
OBAO_CSC_EXP="newsrc_obao_seg_csc12_s0"
OBAO_ACDC_EXP="newsrc_obao_seg_acdc4_s0"

CTAOD_CFG="${CFG_ROOT}/newsrc_ctaod_csc.yaml"
OBAO_CSC_CFG="${CFG_ROOT}/obao_seg_pfn_R_50_CS_C.yaml"
OBAO_ACDC_CFG="${CFG_ROOT}/obao_seg_pfn_R_50_ACDC.yaml"

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
       brightness contrast elastic_transform pixelate jpeg_compression)
CSC5=(fog motion_blur snow brightness defocus_blur)
ACDC4=(fog night rain snow)

make_stream() {
    local suffix="$1"
    shift
    local item stream="("
    for item in "$@"; do
        stream+="\"${item}${suffix}\","
    done
    printf '%s' "${stream%,})"
}

repeat_items() {
    local rounds="$1"
    shift
    local round item
    for ((round=0; round<rounds; round++)); do
        for item in "$@"; do
            printf '%s\n' "$item"
        done
    done
}

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

for path in \
    "${ROOT}/scripts/run_mixed_lt_local.sh" \
    "${ROOT}/scripts/run_ctcr_acdc_local.sh" \
    "${ROOT}/${CTAOD_CFG}" \
    "${ROOT}/${OBAO_CSC_CFG}" \
    "${ROOT}/${OBAO_ACDC_CFG}" \
    "${CHECKPOINT}"; do
    [[ -e "$path" ]] || fail "missing required path: $path"
done

actual_md5="$(md5sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_md5}" == "${EXPECTED_MD5}" ]] ||
    fail "canonical checkpoint MD5 mismatch: ${actual_md5}"

for cfg in "${CTAOD_CFG}" "${OBAO_CSC_CFG}" "${OBAO_ACDC_CFG}"; do
    grep -Fq "/workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" \
        "${ROOT}/${cfg}" ||
        fail "${cfg} does not resolve the canonical checkpoint"
done

declare -a experiments=("${CTAOD_EXP}" "${OBAO_CSC_EXP}" "${OBAO_ACDC_EXP}")
for exp in "${experiments[@]}"; do
    [[ ! -e "${LOG_ROOT}/${exp}.log" ]] ||
        fail "log already exists; refusing overwrite: ${LOG_ROOT}/${exp}.log"
    [[ ! -e "${HOST_OUT}/ctta_acdc/${exp}" ]] ||
        fail "output already exists; refusing overwrite: ${HOST_OUT}/ctta_acdc/${exp}"
done

mkdir -p "${META_ROOT}" "${LOG_ROOT}" "${PATCH_ROOT}"

git_sha="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    git_state=dirty
else
    git_state=clean
fi

stamp="$(date +%Y%m%dT%H%M%S)"
git -C "${ROOT}" status --short > "${PATCH_ROOT}/remaining_ctta_${stamp}.status"
git -C "${ROOT}" diff --binary > "${PATCH_ROOT}/remaining_ctta_${stamp}.patch"

mapfile -t csc_lt_items < <(repeat_items 10 "${CSC5[@]}")
CTAOD_STREAM="$(make_stream "" "${csc_lt_items[@]}")"
OBAO_CSC_STREAM="$(make_stream "_semseg" "${CSC12[@]}")"
acdc_items=()
for item in "${ACDC4[@]}"; do acdc_items+=("acdc_${item}"); done
OBAO_ACDC_STREAM="$(make_stream "_semseg" "${acdc_items[@]}")"

write_meta_start() {
    local exp="$1" method="$2" protocol="$3" gpu="$4" cfg="$5" stream="$6" expected="$7"
    cat > "${META_ROOT}/${exp}.txt" <<META
experiment=${exp}
method=${method}
protocol=${protocol}
gpu=${gpu}
seed=${SEED}
config=${cfg}
git_sha=${git_sha}
git_state=${git_state}
model_weights=/workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth
checkpoint_md5=${actual_md5}
stream=${stream}
expected_evals=${expected}
start_time=$(date -Is)
status=RUNNING
META
}

finish_meta() {
    local exp="$1" rc="$2" expected="$3"
    local log="${LOG_ROOT}/${exp}.log" got errors status
    got="$(grep -c 'in csv format' "${log}" 2>/dev/null || true)"
    errors="$(grep -Ec 'Traceback|CUDA out of memory|^RuntimeError:|^ERROR:' "${log}" 2>/dev/null || true)"
    if ((rc == 0 && got == expected && errors == 0)); then
        status=COMPLETE
    elif ((rc == 0)); then
        status=INCOMPLETE
    else
        status=FAILED
    fi
    {
        echo "end_time=$(date -Is)"
        echo "exit_code=${rc}"
        echo "eval_count=${got}"
        echo "error_count=${errors}"
        echo "final_status=${status}"
    } >> "${META_ROOT}/${exp}.txt"
    echo "[${status}] ${exp}: ${got}/${expected} evaluations, exit=${rc}, errors=${errors}"
    [[ "${status}" == COMPLETE ]]
}

run_job() {
    local gpu="$1" exp="$2" method="$3" protocol="$4" cfg="$5" stream="$6" expected="$7"
    local rc
    write_meta_start "${exp}" "${method}" "${protocol}" "${gpu}" "${cfg}" "${stream}" "${expected}"
    echo "[$(date -Is)] GPU ${gpu} START ${exp}"
    set +e
    if [[ "${protocol}" == CSC* ]]; then
        STREAM="${stream}" bash "${ROOT}/scripts/run_mixed_lt_local.sh" \
            "${gpu}" "${exp}" "${cfg}" "${SEED}"
        rc=$?
    else
        STREAM="${stream}" bash "${ROOT}/scripts/run_ctcr_acdc_local.sh" \
            "${gpu}" "${exp}" "${cfg}" "${SEED}"
        rc=$?
    fi
    set -e
    finish_meta "${exp}" "${rc}" "${expected}"
}

worker_gpu0() {
    exec 9>"/tmp/amrod_gpu0.lock"
    flock -n 9 || fail "GPU 0 batch lock is held"
    run_job 0 "${OBAO_CSC_EXP}" "OBAO-seg" "CSC12" \
        "${OBAO_CSC_CFG}" "${OBAO_CSC_STREAM}" 12
}

worker_gpu1() {
    exec 9>"/tmp/amrod_gpu1.lock"
    flock -n 9 || fail "GPU 1 batch lock is held"
    run_job 1 "${CTAOD_EXP}" "CTAOD" "CSC-LT" \
        "${CTAOD_CFG}" "${CTAOD_STREAM}" 50
    run_job 1 "${OBAO_ACDC_EXP}" "OBAO-seg" "ACDC4" \
        "${OBAO_ACDC_CFG}" "${OBAO_ACDC_STREAM}" 4
}

echo "Canonical checkpoint MD5: ${actual_md5}"
echo "Git: ${git_sha} (${git_state})"
echo "GPU 0: ${OBAO_CSC_EXP}"
echo "GPU 1: ${CTAOD_EXP}, then ${OBAO_ACDC_EXP}"

if [[ "${DRY_RUN}" == 1 ]]; then
    echo "DRY_RUN=1: preflight passed; no experiments launched."
    exit 0
fi

worker_gpu0 &
pid0=$!
worker_gpu1 &
pid1=$!

set +e
wait "${pid0}"
rc0=$?
wait "${pid1}"
rc1=$?
set -e

if ((rc0 != 0 || rc1 != 0)); then
    echo "BATCH FAILED: gpu0=${rc0}, gpu1=${rc1}" >&2
    exit 1
fi

echo "ALL REMAINING CTAOD/OBAO EXPERIMENTS COMPLETE"
