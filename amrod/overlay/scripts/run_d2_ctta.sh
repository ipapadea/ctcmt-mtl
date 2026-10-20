#!/usr/bin/env bash
# Self-service detectron2 CTTA runner (Cityscapes source models).
#
# Run any combination of method × benchmark × repeats with one command.
# Results are saved to /data/ilias/panoptic_fpn/output/ctta_<bench>/<method>/
# and can be harvested with:
#   python3 tools/harvest_results.py --print
#
# USAGE
# -----
#   bash scripts/run_d2_ctta.sh --gpu GPU --method METHOD --bench BENCH [OPTIONS]
#
# ARGUMENTS
#   --gpu    GPU      GPU index (default: 0)
#   --method METHOD   Adaptation method:
#                       amrod        AMROD, our trained Mask R-CNN source
#                       amrod-off    AMROD, official paper source weights (cityscapes_train_final.pth)
#                       ctcmt-det    CT-CMT det-only, Mask R-CNN source
#                       ctcmt-seg    CT-CMT seg-only, Panoptic FPN source
#                       ctcmt-mtl    CT-CMT MTL v4b, Panoptic FPN source
#                       cotta        CoTTA seg, Panoptic FPN source
#   --bench BENCH     Benchmark:
#                       acdc         Cityscapes → ACDC (fog/night/rain/snow), 1 or 10 rounds (Table 5)
#                       cs-c         Cityscapes → Cityscapes-C (12 corruptions, 1 round)  (Table 2)
#                       cs-c-lt      Cityscapes → Cityscapes-C (5 corruptions × 10 rounds) (Table 3)
#                       shift        Cityscapes → SHIFT (4 conditions, 1 round)            (Table 4)
#                       foggy        Cityscapes → Foggy Cityscapes
#   --repeats N       CTTA rounds: default 1 for all benches except cs-c-lt which defaults to 10.
#                     Explicit --repeats N always overrides the default (use for smoke tests).
#
# EXAMPLES
#   # CT-CMT-MTL on ACDC, GPU 2
#   bash scripts/run_d2_ctta.sh --gpu 2 --method ctcmt-mtl --bench acdc
#
#   # AMROD on Cityscapes-C, GPU 0
#   bash scripts/run_d2_ctta.sh --gpu 0 --method amrod --bench cs-c
#
#   # CT-CMT-Det long-term ACDC (10 rounds), GPU 1
#   bash scripts/run_d2_ctta.sh --gpu 1 --method ctcmt-det --bench acdc --repeats 10
#
#   # All methods on ACDC (sequential, one GPU)
#   for m in amrod ctcmt-det ctcmt-seg ctcmt-mtl cotta; do
#     bash scripts/run_d2_ctta.sh --gpu 0 --method $m --bench acdc
#   done

set -euo pipefail

GPU=0
METHOD=""
BENCH="acdc"
REPEATS=1
REPEATS_EXPLICIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU="$2";     shift 2 ;;
        --method)  METHOD="$2";  shift 2 ;;
        --bench)   BENCH="$2";   shift 2 ;;
        --repeats) REPEATS="$2"; REPEATS_EXPLICIT=true; shift 2 ;;
        -h|--help) head -60 "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$METHOD" ]]; then
    echo "ERROR: --method is required."; exit 1
fi

# Table 3 is intrinsically a 10-round long-term protocol.
# Keep explicit --repeats values available for smoke tests/debugging.
if [[ "$BENCH" == "cs-c-lt" && "$REPEATS_EXPLICIT" == false ]]; then
    REPEATS=10
fi

# ---- Method → run_ctta_acdc.sh TRACK ----
case "${METHOD}__${BENCH}" in
    amrod__acdc)        TRACK="amrod" ;;
    amrod-off__acdc)    TRACK="amrod_official" ;;   # official paper weights
    ctcmt-det__acdc)    TRACK="ctcmt_det" ;;
    ctcmt-seg__acdc)    TRACK="ctcmt_seg" ;;
    ctcmt-mtl__acdc)    TRACK="ctcmt_v4b" ;;
    cotta__acdc)        TRACK="cotta_semseg" ;;
    amrod__cs-c)        TRACK="amrod_cs_c" ;;
    amrod-off__cs-c)    TRACK="amrod_official_cs_c" ;;
    ctcmt-det__cs-c)    TRACK="ctcmt_det_mr_cs_c" ;;
    ctcmt-seg__cs-c)    TRACK="ctcmt_seg_cs_c" ;;
    ctcmt-mtl__cs-c)    TRACK="ctcmt_mtl_cs_c" ;;
    cotta__cs-c)        TRACK="cotta_cs_c" ;;
    # CS-C long-term (Table 3): 5 corruptions × REPEATS rounds
    amrod__cs-c-lt)     TRACK="amrod_cs_c_lt" ;;
    amrod-off__cs-c-lt) TRACK="amrod_official_cs_c_lt" ;;
    ctcmt-det__cs-c-lt) TRACK="ctcmt_det_mr_cs_c_lt" ;;
    ctcmt-seg__cs-c-lt) TRACK="ctcmt_seg_cs_c_lt" ;;
    ctcmt-mtl__cs-c-lt) TRACK="ctcmt_mtl_cs_c_lt" ;;
    amrod__shift)       TRACK="amrod_shift" ;;
    ctcmt-det__shift)   TRACK="ctcmt_det_shift" ;;
    amrod__foggy)       TRACK="amrod_foggy" ;;
    amrod-off__foggy)   TRACK="amrod_foggy" ;;   # same arch; official weights don't have a foggy variant
    ctcmt-det__foggy)   TRACK="ctcmt_det_mr_foggy" ;;
    ctcmt-mtl__foggy)   TRACK="ctcmt_v4b_foggy_mtl" ;;   # best MTL config
    *)
        echo "ERROR: unsupported method='$METHOD' + bench='$BENCH' combination."
        echo "Supported: amrod/amrod-off/ctcmt-det/ctcmt-seg/ctcmt-mtl/cotta  ×  acdc/cs-c/cs-c-lt/shift/foggy"
        exit 1
        ;;
esac

LOG="/tmp/d2_${TRACK}_x${REPEATS}.log"
echo "========================================"
echo "  D2 CTTA"
echo "  GPU:     ${GPU}"
echo "  Method:  ${METHOD}  →  track=${TRACK}"
echo "  Bench:   ${BENCH}"
echo "  Repeats: ${REPEATS}"
echo "  Log:     ${LOG}"
echo "========================================"

bash /home/ilias/AMROD/scripts/run_ctta_acdc.sh "$GPU" "$TRACK" "$REPEATS" \
  2>&1 | tee "$LOG"

echo ""
echo "[done] Harvest: python3 /home/ilias/AMROD/tools/harvest_results.py --print"
