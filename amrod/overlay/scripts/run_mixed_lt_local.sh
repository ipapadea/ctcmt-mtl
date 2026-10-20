#!/usr/bin/env bash
set -euo pipefail

# Cityscapes-C mixed long-term stream on this machine:
#   (fog -> motion_blur -> snow -> brightness -> defocus_blur) x 10, no reset.
#
#   bash scripts/run_mixed_lt_local.sh GPU EXP_NAME CONFIG [SEED]

GPU="${1:?GPU id required}"
EXP="${2:?experiment name required}"
CFG="${3:?config required}"
SEED="${4:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
# Mount whole: annotations/*.json are relative symlinks into this parent.
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

# Optional diagnostics. Unset by default, so the docker invocation and the
# adaptation trajectory are unchanged for every existing run.
DIAG_ENVS=()
[[ -n "${CTCMT_DIAG_JSONL:-}" ]] && DIAG_ENVS+=(-e "CTCMT_DIAG_JSONL=${CTCMT_DIAG_JSONL}")
[[ -n "${CTCMT_MASK_DIAG_GT_ROOT:-}" ]] && DIAG_ENVS+=(-e "CTCMT_MASK_DIAG_GT_ROOT=${CTCMT_MASK_DIAG_GT_ROOT}")
[[ -n "${CTCMT_MASK_DIAG_THRESHOLDS:-}" ]] && DIAG_ENVS+=(-e "CTCMT_MASK_DIAG_THRESHOLDS=${CTCMT_MASK_DIAG_THRESHOLDS}")

CYCLE=(fog motion_blur snow brightness defocus_blur)
OUT="/workspace/output/ctta_acdc/${EXP}"
LOG="${HOST_OUT}/logs/${EXP}.log"
mkdir -p "$(dirname "${LOG}")"

STREAM="${STREAM:-}"
if [[ -z "${STREAM}" ]]; then
  STREAM="("
  for _ in $(seq 1 10); do for c in "${CYCLE[@]}"; do STREAM+="\"${c}_mtl\","; done; done
  STREAM="${STREAM%,})"
fi

for c in "${CYCLE[@]}"; do
  [[ -d "${CSC_ROOT}/${c}/leftImg8bit/val" ]] || { echo "ERROR: missing ${CSC_ROOT}/${c}/leftImg8bit/val" >&2; exit 2; }
done

# Mount every corruption directory that has validation images, not only the
# 5-domain default CYCLE. A custom STREAM (e.g. the 12-corruption short-term
# protocol) references corruptions outside CYCLE; if those are not mounted the
# run dies in a dataloader worker with a missing-file error on the 2nd domain.
# 'cityscapes' is skipped: /datasets/cityscapes is already bound to CITYSCAPES_ROOT.
MOUNTS=()
for d in "${CSC_ROOT}"/*/; do
  c="$(basename "${d}")"
  [[ "${c}" == "cityscapes" ]] && continue
  [[ -d "${d}/leftImg8bit/val" ]] || continue
  MOUNTS+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
done
[[ -f "${CITYSCAPES_ROOT}/annotations/instancesonly_filtered_gtFine_val.json" ]] || { echo "ERROR: missing detection GT" >&2; exit 2; }

echo "======================================================================"
echo "CITYSCAPES-C MIXED LONG-TERM x10 (50 evals, no reset)"
echo "EXP    : ${EXP}   GPU: ${GPU}   SEED: ${SEED}"
echo "CONFIG : ${CFG}"
echo "======================================================================"

rm -rf "${HOST_OUT}/ctta_acdc/${EXP}"

docker run --rm \
  --gpus "\"device=${GPU}\"" \
  --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e NVIDIA_TF32_OVERRIDE=0 \
  -e DETECTRON2_DATASETS=/datasets \
  -e PYTHONPATH=/workspace/amrod/detectron2 \
  ${DIAG_ENVS[@]+"${DIAG_ENVS[@]}"} \
  -v "${HOST_REPO}:/workspace/amrod:ro" \
  -v "${CITYSCAPES_ROOT}:/datasets/cityscapes:ro" \
  "${MOUNTS[@]}" \
  -v "${HOST_OUT}:/workspace/output" \
  -w /workspace/amrod \
  "${DOCKER_IMAGE}" bash -c "
    python detectron2/tools/train_net.py \
      --config-file ${CFG} \
      --eval-only \
      --num-gpus 1 \
      OUTPUT_DIR ${OUT} \
      SEED ${SEED} \
      DATASETS.TEST '${STREAM}' ${EXTRA_OPTS}
  " 2>&1 | tee "${LOG}"

echo
echo "DONE: ${EXP}"
echo "LOG : ${LOG}"
