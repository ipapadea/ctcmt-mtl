#!/usr/bin/env bash
set -Eeuo pipefail

# DAT-BF (same-source Panoptic-FPN port) launcher: smoke test or full seed run.
#
#   bash scripts/dat_bf_run.sh smoke GPU csc12_defocus_first100        # first 100 frames of CSC12 defocus_blur
#   bash scripts/dat_bf_run.sh full  GPU newsrc_dat_bf_csc12_s0 CSC12 0
#   bash scripts/dat_bf_run.sh full  GPU newsrc_dat_bf_acdc4_s0 ACDC4 0
#   bash scripts/dat_bf_run.sh full  GPU newsrc_dat_bf_cscLT_s0  CSCLT 0   # (fog,motion_blur,snow,brightness,defocus_blur) x10
#   bash scripts/dat_bf_run.sh full  GPU newsrc_dat_bf_acdcLT_s0 ACDCLT 0  # (fog,night,rain,snow) x10
#   DRY_RUN=1 ...   preflight only (no docker run)
#
# Differences from the older run_*_local.sh helpers: this launcher NEVER deletes or reuses
# an existing run directory, refuses to run on an unexpectedly dirty tree, verifies the
# checkpoint MD5 and records provenance before any GPU work. Nothing is reset between
# frames and no domain-boundary information is passed to the model (the model only sees
# the image stream).

MODE="${1:?mode required: smoke|full}"
GPU="${2:?GPU id required}"
EXP="${3:?experiment name required}"
PROTOCOL="${4:-CSC12}"
SEED="${5:-0}"

HOST_REPO="${HOST_REPO:-/media/ilias/DATA/ilias/AMROD}"
HOST_OUT="${HOST_OUT:-/media/ilias/DATA/ilias/amrod_output}"
CSC_ROOT="${CSC_ROOT:-/media/ilias/DATA/ilias/cityscapes_c}"
CITYSCAPES_ROOT_CSC="${CITYSCAPES_ROOT_CSC:-/media/ilias/DATA/ilias/cityscapes}"
ACDC_ROOT="${ACDC_ROOT:-/media/ilias/DATA/ilias/acdc}"
CITYSCAPES_ROOT_ACDC="${CITYSCAPES_ROOT_ACDC:-/media/ilias/DATA/ilias/cityscapes_pfn}"
CITYSCAPES_ANN_ROOT="${CITYSCAPES_ANN_ROOT:-/media/ilias/DATA/ilias/cityscapes/annotations}"
DOCKER_IMAGE="${DOCKER_IMAGE:-amrod:latest}"
DRY_RUN="${DRY_RUN:-0}"

CHECKPOINT="${HOST_OUT}/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
EXPECTED_MD5="5f63c7578cb925552fd1b9d69980937a"
DAT_BRANCH="dat-bf-panoptic-fpn"
# Algorithm commit locked for all DAT-BF experiments. Tooling (scripts/, tools/ scorer) may move on,
# but the model, config defaults, configs and shared train_net must stay byte-identical to it.
ALGO_COMMIT="9837586bd59ca0a8aa65ff9247a2675e944b4468"
ALGO_PATHS=(detectron2/detectron2 detectron2/configs detectron2/tools/train_net.py)
SNAPSHOT_TAG="pre-dat-bf-20261013"
# Untracked files that are known and intentionally excluded from the snapshot (Decision 8).
ALLOWED_UNTRACKED=(
    "detectron2/detectron2/config/defaults.py.pre_fisher"
    "detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py.pre_fisher"
    "detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py.pre_taskcoco"
    "mtl_papers/"
    "nohup.out"
    "paper26_fixes.diff"
    "papers/"
)

CSC12=(defocus_blur glass_blur motion_blur zoom_blur snow frost fog
       brightness contrast elastic_transform pixelate jpeg_compression)
ACDC4=(fog night rain snow)
CSC_LT_CYCLE=(fog motion_blur snow brightness defocus_blur)   # same cycle as the other baselines' CSC-LT
LT_ROUNDS=10

fail() { echo "ERROR: $*" >&2; exit 2; }

[[ "${MODE}" == smoke || "${MODE}" == full ]] || fail "mode must be smoke or full"
[[ "${PROTOCOL}" == CSC12 || "${PROTOCOL}" == ACDC4 || "${PROTOCOL}" == CSCLT || "${PROTOCOL}" == ACDCLT ]] ||
    fail "protocol must be CSC12, ACDC4, CSCLT or ACDCLT"
[[ "${GPU}" =~ ^[0-9]+$ && "${SEED}" =~ ^[0-9]+$ ]] || fail "GPU and SEED must be integers"
[[ "${EXP}" =~ ^[A-Za-z0-9_]+$ ]] || fail "EXP must match [A-Za-z0-9_]+"

# ---- mode-specific layout: smoke outputs can never collide with paper runs ----
if [[ "${MODE}" == smoke ]]; then
    [[ "${PROTOCOL}" == CSC12 ]] || fail "smoke test is defined for CSC12 only"
    SMOKE_FRAMES=100
    SMOKE_DOMAIN="defocus_blur"
    RUN_REL="dat_bf_smoke/${EXP}"
    LOG="${HOST_OUT}/dat_bf_smoke/${EXP}.log"
    META="${HOST_OUT}/dat_bf_smoke/${EXP}.meta.txt"
    STREAM="(\"${SMOKE_DOMAIN}_semseg_first${SMOKE_FRAMES}\",)"
    SCORE_SPECS="${SMOKE_DOMAIN}_semseg:${SMOKE_FRAMES}"
    EXPECTED_EVALS=1
    CFG="detectron2/configs/Cityscapes/newsrc_dat_bf_csc.yaml"
else
    [[ "${EXP}" != dat_bf_smoke* && "${EXP}" != *smoke* ]] || fail "full-run name must not contain 'smoke'"
    case "${PROTOCOL}" in
        CSC12)
            [[ "${EXP}" == "newsrc_dat_bf_csc12_s${SEED}" ]] || fail "CSC12 full run must be named newsrc_dat_bf_csc12_s${SEED}"
            items=("${CSC12[@]}"); suffix="_semseg"; CFG="detectron2/configs/Cityscapes/newsrc_dat_bf_csc.yaml" ;;
        ACDC4)
            [[ "${EXP}" == "newsrc_dat_bf_acdc4_s${SEED}" ]] || fail "ACDC4 full run must be named newsrc_dat_bf_acdc4_s${SEED}"
            items=("${ACDC4[@]/#/acdc_}"); suffix="_semseg"; CFG="detectron2/configs/Cityscapes/newsrc_dat_bf_acdc.yaml" ;;
        CSCLT)
            [[ "${EXP}" == "newsrc_dat_bf_cscLT_s${SEED}" ]] || fail "CSCLT full run must be named newsrc_dat_bf_cscLT_s${SEED}"
            items=(); for ((r=0; r<LT_ROUNDS; r++)); do items+=("${CSC_LT_CYCLE[@]}"); done
            suffix="_semseg"; CFG="detectron2/configs/Cityscapes/newsrc_dat_bf_csc.yaml" ;;
        ACDCLT)
            [[ "${EXP}" == "newsrc_dat_bf_acdcLT_s${SEED}" ]] || fail "ACDCLT full run must be named newsrc_dat_bf_acdcLT_s${SEED}"
            items=(); for ((r=0; r<LT_ROUNDS; r++)); do items+=("${ACDC4[@]/#/acdc_}"); done
            suffix="_semseg"; CFG="detectron2/configs/Cityscapes/newsrc_dat_bf_acdc.yaml" ;;
    esac
    STREAM="("; SCORE_SPECS=""
    for it in "${items[@]}"; do STREAM+="\"${it}${suffix}\","; SCORE_SPECS+="${it}${suffix} "; done
    STREAM="${STREAM%,})"
    EXPECTED_EVALS="${#items[@]}"
    RUN_REL="ctta_acdc/${EXP}"
    LOG="${HOST_OUT}/logs/${EXP}.log"
    META="${HOST_OUT}/dat_bf_meta/${EXP}.meta.txt"
fi
RUN_HOST="${HOST_OUT}/${RUN_REL}"
RUN_CONT="/workspace/output/${RUN_REL}"

# ---- refuse to overwrite anything ----
for path in "${RUN_HOST}" "${LOG}" "${META}"; do
    [[ ! -e "${path}" ]] || fail "already exists; refusing to overwrite/reuse: ${path}"
done

# ---- source checkpoint ----
[[ -f "${CHECKPOINT}" ]] || fail "missing checkpoint ${CHECKPOINT}"
actual_md5="$(md5sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_md5}" == "${EXPECTED_MD5}" ]] || fail "checkpoint MD5 mismatch: ${actual_md5}"
grep -Fq "/workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth" "${HOST_REPO}/${CFG}" ||
    fail "${CFG} does not resolve the canonical checkpoint"

# ---- git provenance and cleanliness ----
[[ -d "${HOST_REPO}/.git" ]] || fail "${HOST_REPO} is not a git checkout"
git_branch="$(git -C "${HOST_REPO}" rev-parse --abbrev-ref HEAD)"
[[ "${git_branch}" == "${DAT_BRANCH}" ]] || fail "must run from branch ${DAT_BRANCH} (on ${git_branch})"
git_sha="$(git -C "${HOST_REPO}" rev-parse HEAD)"
git -C "${HOST_REPO}" merge-base --is-ancestor "${SNAPSHOT_TAG}" HEAD || fail "${SNAPSHOT_TAG} is not an ancestor of HEAD"
tracked_dirty="$(git -C "${HOST_REPO}" status --porcelain --untracked-files=no)"
[[ -z "${tracked_dirty}" ]] || fail "tracked files modified; commit first:"$'\n'"${tracked_dirty}"
unexpected=""
while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    path="${line#?? }"
    ok=0
    for allowed in "${ALLOWED_UNTRACKED[@]}"; do
        [[ "${path}" == "${allowed}" ]] && ok=1
    done
    ((ok)) || unexpected+="${line}"$'\n'
done < <(git -C "${HOST_REPO}" status --porcelain --untracked-files=normal | grep '^??' || true)
[[ -z "${unexpected}" ]] || fail "unexpected untracked files (not in the Decision-8 allow-list):"$'\n'"${unexpected}"
git -C "${HOST_REPO}" cat-file -e "${ALGO_COMMIT}^{commit}" || fail "locked algorithm commit ${ALGO_COMMIT} not found"
git -C "${HOST_REPO}" merge-base --is-ancestor "${ALGO_COMMIT}" HEAD || fail "${ALGO_COMMIT} is not an ancestor of HEAD"
if ! git -C "${HOST_REPO}" diff --quiet "${ALGO_COMMIT}" HEAD -- "${ALGO_PATHS[@]}"; then
    fail "algorithm paths differ from locked commit ${ALGO_COMMIT}: $(git -C "${HOST_REPO}" diff --stat "${ALGO_COMMIT}" HEAD -- "${ALGO_PATHS[@]}" | tail -1)"
fi
git_state="clean (tracked tree identical to HEAD; only allow-listed untracked files present)"

# ---- datasets ----
if [[ "${PROTOCOL}" == CSC12 || "${PROTOCOL}" == CSCLT ]]; then
    MOUNTS=(-v "${CITYSCAPES_ROOT_CSC}:/datasets/cityscapes:ro")
    for c in "${CSC12[@]}"; do
        [[ -d "${CSC_ROOT}/${c}/leftImg8bit/val" ]] || fail "missing ${CSC_ROOT}/${c}/leftImg8bit/val"
        MOUNTS+=(-v "${CSC_ROOT}/${c}:/datasets/${c}:ro")
    done
else
    for d in "${ACDC_ROOT}/rgb_anon" "${ACDC_ROOT}/gt" "${ACDC_ROOT}/gt_detection" \
             "${CITYSCAPES_ROOT_ACDC}/gtFine" "${CITYSCAPES_ANN_ROOT}"; do
        [[ -e "${d}" ]] || fail "required path not found: ${d}"
    done
    MOUNTS=(-v "${ACDC_ROOT}:/datasets/ACDC:ro" -v "${CITYSCAPES_ROOT_ACDC}:/datasets/cityscapes:ro"
            -v "${CITYSCAPES_ANN_ROOT}:/datasets/annotations:ro")
fi

echo "======================================================================"
echo "DAT-BF (same-source Panoptic-FPN port), mode=${MODE}"
echo "EXP: ${EXP}  PROTOCOL: ${PROTOCOL}  GPU: ${GPU}  SEED: ${SEED}"
echo "CONFIG: ${CFG}"
echo "STREAM: ${STREAM}"
echo "OUTPUT: ${RUN_HOST}"
echo "GIT: ${git_sha} (${git_state})"
echo "ALGORITHM COMMIT (verified identical): ${ALGO_COMMIT}"
echo "MD5: ${actual_md5}"
echo "======================================================================"
if [[ "${DRY_RUN}" == 1 ]]; then
    echo "DRY_RUN=1: preflight passed; nothing launched."
    exit 0
fi

# ---- provenance record (written before any GPU work) ----
mkdir -p "$(dirname "${LOG}")" "$(dirname "${META}")"
{
    echo "experiment=${EXP}"
    echo "mode=${MODE}"
    [[ "${MODE}" == smoke ]] && echo "SMOKE_TEST=1 (never a paper seed run; ${SMOKE_FRAMES} frames of ${SMOKE_DOMAIN})"
    echo "method=DAT-BF (same-source Panoptic-FPN port)"
    echo "protocol=${PROTOCOL}"
    echo "gpu=${GPU}"
    echo "seed=${SEED}"
    echo "config=${CFG}"
    echo "stream=${STREAM}"
    echo "git_branch=${git_branch}"
    echo "git_sha=${git_sha}"
    echo "git_state=${git_state}"
    echo "algorithm_commit=${ALGO_COMMIT} (detectron2/detectron2, detectron2/configs and train_net.py verified identical to HEAD)"
    echo "snapshot_tag=${SNAPSHOT_TAG} -> $(git -C "${HOST_REPO}" rev-parse "${SNAPSHOT_TAG}^{commit}")"
    echo "model_weights=/workspace/output/panoptic_fpn_R50_cityscapes_segw1/model_final.pth"
    echo "checkpoint_md5=${actual_md5}"
    echo "gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null || echo unknown)"
    echo "start_time=$(date -Is)"
    echo "status=RUNNING"
    echo "allowed_untracked:"; git -C "${HOST_REPO}" status --porcelain --untracked-files=normal | grep '^??' | sed 's/^/  /' || true
    echo "resolved_config=${RUN_HOST}/config.yaml (dumped by the run itself)"
} > "${META}"

if [[ "${MODE}" == smoke ]]; then
    RUN_CMD="python detectron2/tools/dat_bf_smoke_entry.py --config-file ${CFG} --eval-only --num-gpus 1 \
        --truncate ${SMOKE_DOMAIN}_semseg:${SMOKE_FRAMES} OUTPUT_DIR ${RUN_CONT} SEED ${SEED} DATASETS.TEST '${STREAM}'"
    SCORE_CMD="python detectron2/tools/dat_bf_score_predictions.py --run-dir ${RUN_CONT} --datasets ${SCORE_SPECS} \
        --online-json ${RUN_CONT}/online_results.json"
else
    RUN_CMD="python detectron2/tools/train_net.py --config-file ${CFG} --eval-only --num-gpus 1 \
        OUTPUT_DIR ${RUN_CONT} SEED ${SEED} DATASETS.TEST '${STREAM}'"
    SCORE_CMD="python detectron2/tools/dat_bf_score_predictions.py --run-dir ${RUN_CONT} --datasets ${SCORE_SPECS} \
        --online-log /workspace/output/logs/${EXP}.log"
fi

set +e
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
    "${MOUNTS[@]}" \
    -v "${HOST_OUT}:/workspace/output" \
    -w /workspace/amrod \
    "${DOCKER_IMAGE}" bash -c "${RUN_CMD}" 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e

echo "[$(date -Is)] run exit=${rc}; offline scoring..."
set +e
if [[ -f "${RUN_HOST}/dat_bf_frames.jsonl" ]]; then
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e DETECTRON2_DATASETS=/datasets \
        -e PYTHONPATH=/workspace/amrod/detectron2 \
        -v "${HOST_REPO}:/workspace/amrod:ro" \
        "${MOUNTS[@]}" \
        -v "${HOST_OUT}:/workspace/output" \
        -w /workspace/amrod \
        "${DOCKER_IMAGE}" bash -c "${SCORE_CMD}" 2>&1 | tee -a "${LOG}"
    score_rc=${PIPESTATUS[0]}
else
    score_rc=99
fi
set -e

got="$(grep -c 'in csv format' "${LOG}" 2>/dev/null || true)"
errors="$(grep -Ec 'Traceback|CUDA out of memory|^RuntimeError:|^ERROR:' "${LOG}" 2>/dev/null || true)"
if [[ -f "${RUN_HOST}/dat_bf_DIVERGED.json" ]]; then
    final=DIVERGED
elif ((rc == 0 && got == EXPECTED_EVALS && errors == 0 && score_rc == 0)); then
    final=COMPLETE
elif ((rc == 0)); then
    final=INCOMPLETE
else
    final=FAILED
fi
{
    echo "end_time=$(date -Is)"
    echo "exit_code=${rc}"
    echo "scorer_exit_code=${score_rc}"
    echo "eval_count=${got}/${EXPECTED_EVALS}"
    echo "error_count=${errors}"
    echo "final_status=${final}"
} >> "${META}"
echo "[${final}] ${EXP}: ${got}/${EXPECTED_EVALS} evaluations, exit=${rc}, scorer=${score_rc}, errors=${errors}"
echo "META: ${META}"
[[ "${final}" == COMPLETE ]]
