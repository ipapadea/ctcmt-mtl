#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
CITYSCAPES_ROOT="${CITYSCAPES_ROOT:-/media/ilias/DATA/ilias/cityscapes}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
DRY_RUN="${DRY_RUN:-0}"

CFGROOT="detectron2/configs/Cityscapes"

CFG_BASE="${CFGROOT}/newsrc_csc_base.yaml"
CFG_OURS="${CFGROOT}/newsrc_csc_ctcr_lambda1.yaml"

CORRUPTIONS=(
  defocus_blur
  glass_blur
  motion_blur
  zoom_blur
  snow
  frost
  fog
  brightness
  contrast
  elastic_transform
  pixelate
  jpeg_compression
)

STREAM="("
MOUNTS=()

for c in "${CORRUPTIONS[@]}"; do
    d="${CSC_ROOT}/${c}/leftImg8bit/val"
    [[ -d "$d" ]] || {
        echo "ERROR: missing $d" >&2
        exit 2
    }

    STREAM+="\"${c}_mtl\","
    MOUNTS+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
done
STREAM="${STREAM%,})"

[[ -f "${HOST_REPO}/${CFG_BASE}" ]] || {
    echo "ERROR: missing ${CFG_BASE}" >&2
    exit 2
}

[[ -f "${HOST_REPO}/${CFG_OURS}" ]] || {
    echo "ERROR: missing ${CFG_OURS}" >&2
    exit 2
}

[[ -f "${HOST_REPO}/detectron2/tools/paper26_dump_predictions.py" ]] || {
    echo "ERROR: missing paper26_dump_predictions.py" >&2
    exit 2
}

[[ -f "${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" ]] || {
    echo "ERROR: missing new-source checkpoint" >&2
    exit 2
}

run_one () {
    local exp="$1"
    local cfg="$2"

    local out_host="${HOST_OUT}/ctta_acdc/${exp}"
    local out_docker="/workspace/output/ctta_acdc/${exp}"
    local log="${HOST_OUT}/logs/${exp}.log"

    echo "================================================================"
    echo "EXP    : ${exp}"
    echo "GPU    : ${GPU}"
    echo "SEED   : 0"
    echo "CONFIG : ${cfg}"
    echo "STREAM : ${STREAM}"
    echo "================================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    # Never overwrite a previous dump.
    if [[ -e "$out_host" || -e "$log" ]]; then
        echo "ERROR: existing output/log for ${exp}; refusing overwrite." >&2
        exit 3
    fi

    mkdir -p "${HOST_OUT}/logs"

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
      "${DOCKER_IMAGE}" bash -c "
        python detectron2/tools/paper26_dump_predictions.py \
          --config-file ${cfg} \
          --eval-only \
          --num-gpus 1 \
          OUTPUT_DIR ${out_docker} \
          SEED 0 \
          DATASETS.TEST '${STREAM}'
      " 2>&1 | tee "$log"

    evals="$(grep -c 'Evaluation results for .* in csv format:' "$log" || true)"

    if [[ "$evals" -ne 12 ]]; then
        echo "ERROR: ${exp}: ${evals}/12 evaluations" >&2
        exit 4
    fi

    # Verify that predictions survived for every corruption.
    for c in "${CORRUPTIONS[@]}"; do
        d="${out_host}/inference_by_dataset/${c}_mtl"

        [[ -f "${d}/sem_seg_predictions.json" ]] || {
            echo "ERROR: missing semseg dump: $d" >&2
            exit 5
        }

        [[ -f "${d}/instances_predictions.pth" ]] || {
            echo "ERROR: missing detection dump: $d" >&2
            exit 5
        }
    done

    echo "[COMPLETE] ${exp}: 12/12 datasets + prediction dumps"
}


run_one \
  "paper26_qualdump_csc12_base_s0" \
  "${CFG_BASE}"

run_one \
  "paper26_qualdump_csc12_ours_l1_s0" \
  "${CFG_OURS}"

echo
echo "QUALITATIVE CSC12 DUMP COMPLETE"
