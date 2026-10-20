#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

RESULT_ROOT="${RESULT_ROOT:-${HOST_OUT}/paper26_efficiency}"
LOG_ROOT="${LOG_ROOT:-${HOST_OUT}/logs/paper26_efficiency}"
CANONICAL_ROOT="${HOST_OUT}/paper26_efficiency"

DATASET="${DATASET:-defocus_blur_mtl}"
WARMUP="${WARMUP:-50}"
MEASURE="${MEASURE:-400}"
REPEATS="${REPEATS:-3}"

# Default = the original four-method table (already complete, so a bare relaunch
# refuses to overwrite). The SOTA baselines are run with:
#   METHODS="amrod cotta tent" bash scripts/paper26_batch_efficiency.sh 1
METHODS="${METHODS:-base ctcr taskcoco ours}"

# Short/odd-length or partial runs must never land in the canonical results.
if [[ "${RESULT_ROOT}" == "${CANONICAL_ROOT}" ]]; then
    if [[ "${WARMUP}" != 50 || "${MEASURE}" != 400 || "${REPEATS}" != 3 ]]; then
        echo "ERROR: WARMUP/MEASURE/REPEATS differ from 50/400/3; set RESULT_ROOT/LOG_ROOT to a scratch dir."
        exit 2
    fi
fi
[[ "${RESULT_ROOT}" == "${HOST_OUT}/"* ]] || { echo "ERROR: RESULT_ROOT must live under ${HOST_OUT}"; exit 2; }
[[ "${REPEATS}" =~ ^[1-3]$ ]] || { echo "ERROR: REPEATS must be 1, 2 or 3"; exit 2; }
CONTAINER_RESULT_ROOT="/workspace/output/${RESULT_ROOT#"${HOST_OUT}"/}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

# ------------------------------------------------------------
# Safety: final timings should run while the machine has NO
# other CUDA compute processes on either GPU.
# ------------------------------------------------------------
if ! ACTIVE_PIDS="$(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits \
      2>/dev/null
)"; then
    echo "ERROR: nvidia-smi query failed."
    exit 2
fi

if grep -Eq '[0-9]' <<< "${ACTIVE_PIDS}"; then
    echo "ERROR: another CUDA compute process is active."
    echo "For final paper timing, wait until BOTH GPUs are idle."
    echo
    nvidia-smi
    exit 2
fi
echo "=============================================================="
echo "PAPER26 EFFICIENCY BENCHMARK"
echo "GPU      : ${GPU}"
echo "DATASET  : ${DATASET} (multi-task methods; SOTA baselines: per-method view, see below)"
echo "WARMUP   : ${WARMUP}"
echo "MEASURE  : ${MEASURE}"
echo "REPEATS  : ${REPEATS}"
echo "METHODS  : ${METHODS}"
echo "RESULTS  : ${RESULT_ROOT}"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv
echo

declare -A CFG
CFG[base]="detectron2/configs/Cityscapes/newsrc_csc_base.yaml"
CFG[ctcr]="detectron2/configs/Cityscapes/newsrc_csc_ctcr.yaml"
CFG[taskcoco]="detectron2/configs/Cityscapes/newsrc_csc_taskcoco.yaml"
CFG[ours]="detectron2/configs/Cityscapes/newsrc_csc_ctcr_lambda1.yaml"
CFG[amrod]="detectron2/configs/Cityscapes/amrod_newsrc_csc.yaml"
CFG[cotta]="detectron2/configs/Cityscapes/cotta_newsrc_csc.yaml"
CFG[tent]="detectron2/configs/Cityscapes/tent_newsrc_csc.yaml"

# Diagnostic CTCR is deliberately included:
# Base / Task-CoCo / Ours are the paper comparison.
# plain CTCR tells us the implementation overhead of lambda=1 routing.
declare -A LABEL
LABEL[base]="Base"
LABEL[ctcr]="+CTCR"
LABEL[taskcoco]="Task-CoCo"
LABEL[ours]="Ours-lambda1"
LABEL[amrod]="AMROD"
LABEL[cotta]="CoTTA"
LABEL[tent]="TENT"

# Each SOTA baseline runs on the dataset view its meta-architecture was
# validated with (same Cityscapes-C defocus_blur images, same mounts).
declare -A DSET
DSET[base]="${DATASET}"
DSET[ctcr]="${DATASET}"
DSET[taskcoco]="${DATASET}"
DSET[ours]="${DATASET}"
DSET[amrod]="defocus_blur"
DSET[cotta]="defocus_blur_semseg"
DSET[tent]="defocus_blur_semseg"

# Single-task and multi-task methods are different workloads.
declare -A SCOPE
SCOPE[base]="Detection + Segmentation"
SCOPE[ctcr]="Detection + Segmentation"
SCOPE[taskcoco]="Detection + Segmentation"
SCOPE[ours]="Detection + Segmentation"
SCOPE[amrod]="Detection"
SCOPE[cotta]="Segmentation"
SCOPE[tent]="Segmentation"


run_one () {
    local method="$1"
    local rep="$2"

    local cfg="${CFG[$method]}"
    local label="${LABEL[$method]}"
    local dataset="${DSET[$method]}"
    local scope="${SCOPE[$method]}"

    local json="${RESULT_ROOT}/${method}_r${rep}.json"
    local log="${LOG_ROOT}/${method}_r${rep}.log"
    local runtime="${CONTAINER_RESULT_ROOT}/runtime_${method}_r${rep}"

    if [[ -e "${json}" || -e "${log}" ]]; then
        echo "ERROR: existing result for ${method} repeat ${rep}"
        echo "  ${json}"
        echo "  ${log}"
        echo "Refusing overwrite."
        exit 3
    fi

    echo
    echo "--------------------------------------------------------------"
    echo "METHOD : ${label}"
    echo "SCOPE  : ${scope}"
    echo "DATASET: ${dataset}"
    echo "REPEAT : ${rep}/${REPEATS}"
    echo "CONFIG : ${cfg}"
    echo "--------------------------------------------------------------"

    docker run --rm \
      --gpus "\"device=${GPU}\"" \
      --shm-size=8g \
      --user "$(id -u):$(id -g)" \
      -e HOME=/tmp \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -e NVIDIA_TF32_OVERRIDE=0 \
      -e DETECTRON2_DATASETS=/datasets \
      -e PYTHONPATH=/workspace/amrod/detectron2 \
      -v "${HOST_REPO}:/workspace/amrod:ro" \
      -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
      -v "${CSC_ROOT}/defocus_blur:/datasets/defocus_blur:ro" \
      -v "${HOST_OUT}:/workspace/output" \
      -w /workspace/amrod \
      "${DOCKER_IMAGE}" \
      python detectron2/tools/paper26_benchmark_efficiency.py \
        --config-file "${cfg}" \
        --method "${label}" \
        --task-scope "${scope}" \
        --dataset "${dataset}" \
        --seed 0 \
        --repeat "${rep}" \
        --warmup "${WARMUP}" \
        --measure "${MEASURE}" \
        --output-dir "${runtime}" \
        --output-json "${CONTAINER_RESULT_ROOT}/${method}_r${rep}.json" \
        2>&1 | tee "${log}"

    [[ -s "${json}" ]] || {
        echo "ERROR: result JSON missing for ${method} r${rep}"
        exit 4
    }

    # Small cooling / allocator / process gap.
    sleep 10
}

# Rotate order between repeats to reduce systematic thermal/order bias.
case "${METHODS}" in
    "base ctcr taskcoco ours")
        ORDER1=(base ctcr taskcoco ours)
        ORDER2=(ours taskcoco ctcr base)
        ORDER3=(taskcoco base ours ctcr)
        ;;
    "amrod cotta tent")
        # Cyclic Latin square: every method takes each position once.
        ORDER1=(amrod cotta tent)
        ORDER2=(tent amrod cotta)
        ORDER3=(cotta tent amrod)
        ;;
    *)
        echo "ERROR: METHODS must be exactly 'base ctcr taskcoco ours' or 'amrod cotta tent' (got '${METHODS}')"
        exit 2
        ;;
esac

for m in "${ORDER1[@]}"; do run_one "$m" 1; done
(( REPEATS >= 2 )) && for m in "${ORDER2[@]}"; do run_one "$m" 2; done
(( REPEATS >= 3 )) && for m in "${ORDER3[@]}"; do run_one "$m" 3; done

echo
echo "=============================================================="
echo "ALL EFFICIENCY RUNS COMPLETE"
echo "=============================================================="
