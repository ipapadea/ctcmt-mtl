#!/usr/bin/env bash
# Loop 12 AMROD Table-2 corruptions through d2_source_only_eval.py inside the AMROD env.
# Usage: ./scripts/eval_d2_cityscapes_c.sh <ckpt_path_in_container> <run_name> [<gpu>]
set -euo pipefail

CKPT="${1:?usage: eval_d2_cityscapes_c.sh <ckpt_path> <run_name> [<gpu>]}"
RUN_NAME="${2:?usage: eval_d2_cityscapes_c.sh <ckpt_path> <run_name> [<gpu>]}"
GPU="${3:-${CUDA_VISIBLE_DEVICES:-0}}"

CORRUPTIONS=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)

D2_DATASETS="/data/vgcmt/datasets/cityscapes_c_amrod"
OUT_ROOT="/data/vgcmt/work_dirs/amrod_d2_pipeline/${RUN_NAME}"
CFG="/workspace/AMROD/detectron2/tools/cfg_cityscapes_c_short.yaml"

echo "[eval_d2_csc] ckpt=${CKPT}"
echo "[eval_d2_csc] gpu=${GPU}  out=${OUT_ROOT}  d2_datasets=${D2_DATASETS}"

HOST_UID=$(id -u) HOST_GID=$(id -g)
export HOST_UID HOST_GID

for c in "${CORRUPTIONS[@]}"; do
    if [[ -f "${OUT_ROOT}/${c}/results.json" ]]; then
        echo "[eval_d2_csc] ---- ${c} (cached) ----"
        python3 -c "import json; r=json.load(open('${OUT_ROOT}/${c}/results.json')); print(f'  mAP={r[\"mAP\"]:.4f} mAP@50={r[\"mAP_50\"]:.4f}')"
        continue
    fi
    echo "[eval_d2_csc] ---- ${c} ----"
    docker compose -f /home/ilias/AMROD/docker-compose.yml run --rm \
        -e CUDA_VISIBLE_DEVICES="${GPU}" \
        -e CITYSCAPES_C_CORRUPTION="${c}" \
        -e DETECTRON2_DATASETS="${D2_DATASETS}" \
        amrod bash -lc "\
            python tools/d2_source_only_eval.py \
                --config-file '${CFG}' \
                --weights '${CKPT}' \
                --out-dir '${OUT_ROOT}'"
done

echo "[eval_d2_csc] aggregating..."
docker compose -f /home/ilias/AMROD/docker-compose.yml run --rm \
    -e DETECTRON2_DATASETS="${D2_DATASETS}" \
    amrod bash -lc "\
        python tools/aggregate_d2_cityscapes_c.py \
            --run-name '${RUN_NAME}' \
            --ckpt '${CKPT}' \
            --work-root '${OUT_ROOT}'"

echo "[eval_d2_csc] done -> ${OUT_ROOT}/summary.json"
