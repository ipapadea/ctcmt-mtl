#!/usr/bin/env bash
# Regenerate Cityscapes-C under AMROD's pinned imagecorruptions/skimage stack.
# Output: /data/vgcmt/datasets/cityscapes_c_amrod/{corruption}/leftImg8bit/val/...
# Also symlinks cityscapes/annotations/ so D2's register_cityscapes_c finds it.
#
# Idempotent + resumable. If all 12 corruption dirs already have 500 imgs,
# does nothing and exits fast.
set -euo pipefail

HOST_UID=$(id -u) HOST_GID=$(id -g)
export HOST_UID HOST_GID

SRC="/data/vgcmt/datasets/cityscapes"
DST="/data/vgcmt/datasets/cityscapes_c_amrod"

mkdir -p "${DST}/cityscapes/annotations"
ln -sfn "${SRC}/annotations/instancesonly_filtered_gtFine_val.json" \
        "${DST}/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"

echo "[corrgen] src=${SRC}  dst=${DST}"
echo "[corrgen] launching in AMROD container (imagecorruptions==1.1.2)..."

docker compose -f /home/ilias/AMROD/docker-compose.yml run --rm \
    amrod bash -lc "\
        python tools/corrgen_amrod.py \
            --src-root '${SRC}' \
            --dst-root '${DST}' \
            --workers 24"

# D2's register_cityscapes_c uses image_root=<corruption_dir> and JSON
# file_name = "<city>/<basename>.png" (no leftImg8bit/val prefix). Our
# corrgen writes to <corruption>/leftImg8bit/val/<city>/, so bridge the
# two layouts with per-city symlinks at the corruption root.
CORRUPTIONS=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)
for c in "${CORRUPTIONS[@]}"; do
    for city in frankfurt lindau munster; do
        src_dir="${DST}/${c}/leftImg8bit/val/${city}"
        dst_link="${DST}/${c}/${city}"
        [ -d "$src_dir" ] && [ ! -e "$dst_link" ] && \
            ln -sfn "leftImg8bit/val/${city}" "$dst_link"
    done
done

echo "[corrgen] done"
du -sh "${DST}"

mkdir -p /data/vgcmt/status
touch /data/vgcmt/status/corrgen.done
echo "[corrgen] touched /data/vgcmt/status/corrgen.done"
