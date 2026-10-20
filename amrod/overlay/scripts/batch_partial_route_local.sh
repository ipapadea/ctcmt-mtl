
#!/usr/bin/env bash
set -euo pipefail

GPU_ACDC="${1:-0}"
GPU_CSC="${2:-1}"

LOGDIR="/media/ilias/DATA/ilias/amrod_output/logs"
mkdir -p "$LOGDIR"

run_acdc() {
    for tag in r25 r50; do
        exp="newsrc_ctcr_${tag}_acdcLT_s0"
        cfg="detectron2/configs/Cityscapes/newsrc_acdc_ctcr_${tag}.yaml"

        echo "Starting $exp on GPU $GPU_ACDC"

        STREAM="$ACDC_STREAM" \
          bash scripts/run_ctcr_acdc_local.sh \
            "$GPU_ACDC" "$exp" "$cfg" 0

        echo "Completed $exp"
    done
}

run_csc() {
    for tag in r25 r50; do
        exp="newsrc_ctcr_${tag}_cscLT_s0"
        cfg="detectron2/configs/Cityscapes/newsrc_csc_ctcr_${tag}.yaml"

        echo "Starting $exp on GPU $GPU_CSC"

        bash scripts/run_mixed_lt_local.sh \
          "$GPU_CSC" "$exp" "$cfg" 0

        echo "Completed $exp"
    done
}

# Construct the exact ACDC long-term stream:
# (fog -> night -> rain -> snow) x 10, without reset.
ACDC_STREAM="("
for _ in $(seq 1 10); do
    for domain in fog night rain snow; do
        ACDC_STREAM+="\"acdc_${domain}_mtl\","
    done
done
ACDC_STREAM="${ACDC_STREAM%,})"

run_acdc > "$LOGDIR/launch_partial_acdc.log" 2>&1 &
pid_acdc=$!

run_csc > "$LOGDIR/launch_partial_csc.log" 2>&1 &
pid_csc=$!

set +e
wait "$pid_acdc"
status_acdc=$?

wait "$pid_csc"
status_csc=$?
set -e

echo "ACDC exit status: $status_acdc"
echo "CSC exit status: $status_csc"

if (( status_acdc != 0 || status_csc != 0 )); then
    exit 1
fi

echo "All four partial-routing experiments completed."
