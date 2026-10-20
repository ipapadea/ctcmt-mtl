#!/usr/bin/env bash
# Batch experiments for the YOLOX-MTL CTCMT adapter (one queue per GPU). NOTHING is started by this file
# other than the queue of the GPU you pass; start one queue per GPU yourself (nohup / tmux), e.g.
#
#   cd <repo>/shift-detection-tta
#   setsid nohup bash docker/batch_mtl.sh dev 0 > $OUT_ROOT/dev_gpu0.out 2>&1 < /dev/null &
#   setsid nohup bash docker/batch_mtl.sh dev 1 > $OUT_ROOT/dev_gpu1.out 2>&1 < /dev/null &
#
#   bash docker/batch_mtl.sh STAGE GPU            (STAGE: plan | dev | dev2 | final)
#
# STAGE=plan   prints the queue of both GPUs and the estimated frames, runs nothing.
# STAGE=dev    stability / lr check on the ACDC *val* split (never used in the benchmark streams).
#                GPU0: dev_lr0 (adapter with lr=0 must reproduce the source model), dev5_lr3e-5
#                GPU1: dev5_lr1e-5
#              ("dev5" = the 406-frame dev stream repeated 5 times, 2030 frames, checks that the lr stays stable)
# STAGE=dev2   lr / gradient-clipping check on the ACDC val stream repeated 10 times (4060 frames, as many rounds as the
#              long-term benchmarks). All with CT-CR 0.3. ~1 h per run.
#                GPU0: dev10_lr3e-5_clip1000, dev10_lr1e-5_clip1000
#                GPU1: dev10_lr3e-5_clip300,  dev10_lr1e-5_clip300
# STAGE=final  needs LR=<value>; optional MAXGN=<clip> (gradient-norm clip of the adapter); SEEDS="0" (default) or
#              e.g. SEEDS="0 42 123".
#                GPU0: CTCMT   (weight_ctcr 0.3) on acdc4, csc12, acdcLT, cscLT
#                GPU1: Base    (weight_ctcr 0)   on acdc4, csc12, acdcLT, cscLT
#              The Source row is not re-run: results/yolox_mtl_final source-only logs (ACDC4, CSC12) already exist,
#              and Source on the long-term streams = the short-term domain results repeated.
#
# Finished runs (log contains "[SEGMEAN") are skipped. An unfinished run (e.g. after a power cut) is moved to
# <name>.incomplete.<timestamp> and restarted from scratch (no mid-run checkpointing).
#
# Env: DATA_HOST (required), CKPT (required), REPO, OUT_ROOT, IMAGE
set -uo pipefail

STAGE="${1:?STAGE: plan|dev|dev2|final}"; GPU="${2:-}"
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export DATA_HOST="${DATA_HOST:?datasets root (contains ACDC/, cityscapes/ and the 12 corruption folders)}"
OUT_ROOT="${OUT_ROOT:-$HOME/work_dirs_mtl}"
export IMAGE="${IMAGE:-shift-detection-tta:cu113}"
CKPT="${CKPT:?path of the YOLOX-MTL v2 source checkpoint (epoch_24.pth)}"
export REPO
mkdir -p "$OUT_ROOT/.torch_cache"
export EXTRA_MOUNTS="-v $OUT_ROOT/.torch_cache:/tmp/.cache/torch"
CFG=configs/continuous/mtl_final
ADAPTER=MeanTeacherYOLOXMTLAdapterCTCMTFinal
SEEDS="${SEEDS:-0}"

# name | config | lr | weight_ctcr | threshold_max | seed | frames | max_grad_norm (empty = none)   (one per line)
queue() {
  local g="$1" s lr="${LR:-}" w proto cfg frames thr clip arm
  case "$STAGE" in
    dev)
      if [[ "$g" == 0 ]]; then
        echo "dev_lr0|dev_acdcval.py|0.0|0.3|0.9|0|406|"
        echo "dev5_lr3e-5|dev_acdcval_x5.py|3e-5|0.3|0.9|0|2030|"
      else
        echo "dev5_lr1e-5|dev_acdcval_x5.py|1e-5|0.3|0.9|0|2030|"
      fi ;;
    dev2)
      if [[ "$g" == 0 ]]; then clip=1000; else clip=300; fi
      for lr in 3e-5 1e-5; do
        echo "dev10_lr${lr}_clip${clip}|dev_acdcval_x10.py|$lr|0.3|0.9|0|4060|$clip"
      done ;;
    final)
      [[ -n "$lr" ]] || { echo "ERROR: STAGE=final needs LR=<value>" >&2; return 1; }
      [[ "$g" == 0 ]] && { w=0.3; arm=ctcmt; } || { w=0.0; arm=base; }
      for s in $SEEDS; do
        for proto in acdc4 csc12 acdcLT cscLT; do
          case "$proto" in
            acdc4)  frames=1600;  thr=0.9 ;;
            csc12)  frames=6000;  thr=0.8 ;;
            acdcLT) frames=16000; thr=0.9 ;;
            cscLT)  frames=25000; thr=0.8 ;;
          esac
          echo "${arm}_${proto}_lr${lr}${MAXGN:+_clip$MAXGN}_s${s}|source_only_${proto}.py|$lr|$w|$thr|$s|$frames|${MAXGN:-}"
        done
      done ;;
  esac
}

if [[ "$STAGE" == plan ]]; then
  for g in 0 1; do
    for st in dev dev2 final; do
      STAGE=$st; echo "== GPU$g  stage=$st ${LR:+LR=$LR}"
      queue "$g" | awk -F'|' '{printf "  %-40s %-26s lr=%-6s ctcr=%s thr_max=%s seed=%s clip=%s frames=%s (~%.1f h at 0.9 s/frame)\n",$1,$2,$3,$4,$5,$6,($8==""?"none":$8),$7,$7*0.9/3600}'
    done
  done
  echo "(final needs LR=..., e.g.  LR=3e-5 bash docker/batch_mtl.sh plan)"
  exit 0
fi

[[ "$GPU" == 0 || "$GPU" == 1 ]] || { echo "ERROR: GPU must be 0 or 1" >&2; exit 2; }
[[ -f "$CKPT" ]] || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 2; }
echo "[batch] stage=$STAGE gpu=$GPU repo=$REPO out=$OUT_ROOT ckpt_md5=$(md5sum "$CKPT" | cut -d' ' -f1)"

while IFS='|' read -r name cfg lr w thr seed frames clip; do
  [[ -n "$name" ]] || continue
  out="$OUT_ROOT/$STAGE/$name"
  if [[ -f "$out/eval.log" ]]; then
    if grep -q '\[SEGMEAN' "$out/eval.log"; then echo "[batch] skip finished $name"; continue; fi
    mv "$out" "$out.incomplete.$(date +%Y%m%d_%H%M%S)"
  fi
  echo "[batch] $(date -Is) start $name (frames=$frames)"
  bash "$REPO/docker/run_mtl.sh" "$GPU" "$CFG/$cfg" "$CKPT" "$out" --cfg-options \
    model.adapter._delete_=True model.adapter.type=$ADAPTER model.adapter.lr="$lr" \
    model.adapter.weight_ctcr="$w" model.adapter.threshold_max="$thr" model.adapter.seed="$seed" \
    ${clip:+model.adapter.max_grad_norm=$clip} \
    || echo "[batch] FAILED $name"
  echo "[batch] $(date -Is) end   $name: $(grep -h '\[MEAN\|\[SEGMEAN' "$out/eval.log" 2>/dev/null | sed 's/.*INFO - //' | tr '\n' ' ')"
done < <(queue "$GPU")
echo "[batch] $(date -Is) queue gpu$GPU done"
