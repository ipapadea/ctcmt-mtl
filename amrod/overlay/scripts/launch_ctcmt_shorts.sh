#!/usr/bin/env bash
set -Eeuo pipefail

cd /media/ilias/DATA/ilias/AMROD

LOGDIR=/media/ilias/DATA/ilias/amrod_output/logs
LAUNCH=/media/ilias/DATA/ilias/amrod_output/short_launch
CFG=detectron2/configs/Cityscapes
RUN="${RUN:-0}"

mkdir -p "$LAUNCH"

# Verify runners, configurations and reference logs.
for file in \
    scripts/run_mixed_lt_local.sh \
    scripts/run_ctcr_acdc_local.sh \
    "$CFG/newsrc_csc_base.yaml" \
    "$CFG/newsrc_csc_ctcr.yaml" \
    "$CFG/newsrc_acdc_base.yaml" \
    "$CFG/newsrc_acdc_ctcr.yaml" \
    "$LOGDIR/newsrc_base_cscLT_s0.log" \
    "$LOGDIR/newsrc_base_acdcLT_s0.log"
do
    [[ -f "$file" ]] || {
        echo "MISSING: $file"
        exit 1
    }
done

for runner in \
    scripts/run_mixed_lt_local.sh \
    scripts/run_ctcr_acdc_local.sh
do
    grep -q 'STREAM' "$runner" || {
        echo "Cannot verify STREAM override in $runner"
        exit 1
    }
done

# Infer the existing dataset-name convention.
suffix_from_log() {
    local file="$1"
    local domain="$2"

    if grep -q "${domain}_mtl" "$file"; then
        printf '_mtl'
    elif grep -q "$domain" "$file"; then
        printf ''
    else
        echo "Cannot identify dataset name in $file" >&2
        return 1
    fi
}

CSC_SUFFIX="$(suffix_from_log \
    "$LOGDIR/newsrc_base_cscLT_s0.log" fog)"

ACDC_SUFFIX="$(suffix_from_log \
    "$LOGDIR/newsrc_base_acdcLT_s0.log" acdc_fog)"

make_stream() {
    local prefix="$1"
    local suffix="$2"
    shift 2

    local items=()
    local domain
    for domain in "$@"; do
        items+=("\"${prefix}${domain}${suffix}\"")
    done

    local IFS=,
    printf '(%s)\n' "${items[*]}"
}

CSC12="$(make_stream "" "$CSC_SUFFIX" \
    defocus_blur glass_blur motion_blur zoom_blur \
    snow frost fog brightness contrast \
    elastic_transform pixelate jpeg_compression)"

ACDC4="$(make_stream "acdc_" "$ACDC_SUFFIX" \
    fog night rain snow)"

echo "Source: panoptic_fpn_R50_cityscapes_segw1"
echo "CSC-12: $CSC12"
echo "ACDC-4: $ACDC4"
echo
echo "Planned:"
echo " GPU 0: newsrc_base_csc12_s0 (12)"
echo " GPU 0: newsrc_ctcr_csc12_s0 (12)"
echo " GPU 1: newsrc_base_acdc4_s0 (4)"
echo " GPU 1: newsrc_ctcr_acdc4_s0 (4)"

if [[ "$RUN" != "1" ]]; then
    echo
    echo "PREFLIGHT ONLY — no experiments launched."
    exit 0
fi

run_one() {
    local gpu="$1"
    local name="$2"
    local cfg="$3"
    local runner="$4"
    local stream="$5"
    local expected="$6"

    local log="$LOGDIR/$name.log"
    local launchlog="$LAUNCH/$name.launch.log"

    if [[ -f "$log" ]]; then
        local count
        count="$(grep -c 'in csv format' "$log" || true)"

        if [[ "$count" -eq "$expected" ]] &&
           ! grep -q 'Traceback (most recent call last)' "$log"; then
            echo "[GPU $gpu] SKIP $name ($count/$expected)"
            return 0
        fi

        echo "[GPU $gpu] STOP: existing incomplete log:"
        echo "$log ($count/$expected)"
        echo "Preserved; no overwrite attempted."
        return 1
    fi

    echo "[GPU $gpu] START $name $(date -Is)"

    STREAM="$stream" bash "$runner" \
        "$gpu" "$name" "$cfg" 0 \
        > "$launchlog" 2>&1 || {
            echo "[GPU $gpu] FAILED $name"
            echo "Inspect $launchlog"
            return 1
        }

    local count
    count="$(grep -c 'in csv format' "$log" || true)"

    if [[ "$count" -ne "$expected" ]] ||
       grep -q 'Traceback (most recent call last)' "$log"; then
        echo "[GPU $gpu] INCOMPLETE $name ($count/$expected)"
        return 1
    fi

    echo "[GPU $gpu] DONE $name ($count/$expected)"
}

run_csc() {
    run_one 0 newsrc_base_csc12_s0 \
        "$CFG/newsrc_csc_base.yaml" \
        scripts/run_mixed_lt_local.sh "$CSC12" 12

    run_one 0 newsrc_ctcr_csc12_s0 \
        "$CFG/newsrc_csc_ctcr.yaml" \
        scripts/run_mixed_lt_local.sh "$CSC12" 12
}

run_acdc() {
    run_one 1 newsrc_base_acdc4_s0 \
        "$CFG/newsrc_acdc_base.yaml" \
        scripts/run_ctcr_acdc_local.sh "$ACDC4" 4

    run_one 1 newsrc_ctcr_acdc4_s0 \
        "$CFG/newsrc_acdc_ctcr.yaml" \
        scripts/run_ctcr_acdc_local.sh "$ACDC4" 4
}

run_csc &
PID0=$!

run_acdc &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

echo
echo "GPU 0 status: $STATUS0"
echo "GPU 1 status: $STATUS1"

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
    exit 1
fi

echo "ALL FOUR SHORT-TERM RUNS COMPLETED."
