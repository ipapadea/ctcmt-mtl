#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
METHOD="${4:?method label required}"
SEED="${5:-0}"

shift 5
EXTRA_ARGS=("$@")

HOST_REPO="${HOST_REPO:-$HOME/AMROD}"
HOST_OUT="${HOST_OUT:-$HOME/amrod_output}"

CSC_ROOT="${CSC_ROOT:-/data/vgcmt/datasets/cityscapes_c_amrod}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/data/vgcmt/datasets/cityscapes}"

DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"

OUT="/workspace/output/retention/${EXP}"
LOG="${HOST_OUT}/logs/${EXP}.log"

mkdir -p "${HOST_OUT}/logs"
rm -rf "${HOST_OUT}/retention/${EXP}"

CYCLE=(fog motion_blur snow brightness defocus_blur)

MOUNTS=()

for d in "${CSC_ROOT}"/*/; do
    c="$(basename "${d}")"

    [[ "${c}" == "cityscapes" ]] && continue
    [[ -d "${d}/leftImg8bit/val" ]] || continue

    MOUNTS+=(
      -v "${CSC_ROOT}/${c}:/datasets/${c}:ro"
    )
done

for c in "${CYCLE[@]}"; do
    [[ -d "${CSC_ROOT}/${c}/leftImg8bit/val" ]] || {
        echo "ERROR: missing ${CSC_ROOT}/${c}/leftImg8bit/val"
        exit 2
    }
done

[[ -f \
"${CITYSCAPES_ROOT}/annotations/instancesonly_filtered_gtFine_val.json" ]] || {
    echo "ERROR: missing Cityscapes detection GT"
    exit 2
}

echo "============================================================"
echo "RETENTION PROBE"
echo "EXP       : ${EXP}"
echo "METHOD    : ${METHOD}"
echo "GPU       : ${GPU}"
echo "SEED      : ${SEED}"
echo "CONFIG    : ${CFG}"
echo "HOST_OUT  : ${HOST_OUT}"
echo "CSC_ROOT  : ${CSC_ROOT}"
echo "============================================================"

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
  "${MOUNTS[@]}" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" \
  python detectron2/tools/retention_probe.py \
    --config-file "${CFG}" \
    --output-dir "${OUT}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
echo "CSV : ${HOST_OUT}/retention/${EXP}/retention.csv"
