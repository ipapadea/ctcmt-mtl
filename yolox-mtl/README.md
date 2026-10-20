# yolox-mtl/: CTCMT on YOLOX-X + semantic head (mmdet)

The final CTCMT method re-implemented for a YOLOX-X detector with a semantic head, plus the single-task method of Moraiti et al. on its own checkpoint.
Built on [panagiotamoraiti/shift-tta](https://github.com/panagiotamoraiti/shift-tta) (`shift-detection-tta`, MIT license of the SHIFT code base).
Our changes to eight upstream files (mostly `__init__.py` registrations) are a patch, everything else is new files; `setup.sh` fetches the upstream commit `a2449e4` and applies both.

```bash
bash setup.sh ~/yolox_mtl          # = panagiotamoraiti/shift-tta@a2449e4 + patches/shift_tta_upstream_changes.patch + overlay/
```

What was built, what differs from the Panoptic-FPN implementation and why: [../docs/yolox_mtl.md](../docs/yolox_mtl.md).

## Environment

Docker, CUDA 11.3 (`docker/Dockerfile.leviathan` = the upstream `docker/Dockerfile` with the CUDA forward-compat libraries removed so that it also runs on a driver 470 machine;
`docker/Dockerfile.leviathan.mt` adds `albumentations==1.0.0` and `mmpretrain==1.0.0rc8`, needed only for the Moraiti adapter). Build **from `shift-detection-tta/`** (the Dockerfile copies `requirements*`):

```bash
cd ~/yolox_mtl/shift-detection-tta
docker build -f docker/Dockerfile.leviathan    -t shift-detection-tta:cu113    .
docker build -f docker/Dockerfile.leviathan.mt -t shift-detection-tta:cu113-mt .   # only for configs/continuous/mean_teacher_adapter_yolox/moraiti_*.py
```

## Data and checkpoints

`DATA_HOST` is mounted read-only at `/workspace/data` and must contain

```
ACDC/{rgb_anon, gt, gt_detection}/...              # official ACDC; detection jsons as in AMROD
cityscapes/{leftImg8bit, gtFine}/val/...           # + instancesonly_filtered_gtFine_val.json (8-class COCO json), gtFine *_labelTrainIds.png
<corruption>/leftImg8bit/val/<city>/...            # the 12 Cityscapes-C corruptions (severity 5), same file names as clean val
```

`gt/**/*_gt_labelTrainIds.png` of ACDC and `gtFine/**/*_gtFine_labelTrainIds.png` of Cityscapes are the 19-class masks (`docs/mtl_extension.md` explains how to create them).
If `DATA_HOST` contains symlinks, pass the link targets with `EXTRA_MOUNTS="-v target:target:ro"`.

Source model (not included): `tools/train.py configs/source/yolox_mtl/amp_yolox_x_mtl_v2_8xb4-24e_cityscapes.py` (24 epochs, v2 recipe; see the docs).
The Moraiti et al. checkpoint of the single-task comparison is not included either.

## Running

One run = one `tools/test.py` call inside docker. The adapter is selected and configured on the command line (the stream configs have `adapter=None`, i.e. Source):

```bash
export DATA_HOST=/path/to/datasets IMAGE=shift-detection-tta:cu113
CKPT=/path/to/yolox_x_cs_mtl_v2_amp/epoch_24.pth

# Source (no adaptation) on ACDC4: AP50 / mAP per domain and mIoU per domain
bash docker/run_mtl.sh 0 configs/continuous/mtl_final/source_only_acdc4.py $CKPT out/src_acdc4

# CTCMT (CT-CR 0.3) on ACDC4;   Base = weight_ctcr=0.0.   threshold_max: 0.9 ACDC, 0.8 Cityscapes-C
bash docker/run_mtl.sh 0 configs/continuous/mtl_final/source_only_acdc4.py $CKPT out/ctcmt_acdc4 --cfg-options \
  model.adapter._delete_=True model.adapter.type=MeanTeacherYOLOXMTLAdapterCTCMTFinal \
  model.adapter.lr=3e-5 model.adapter.max_grad_norm=1000 model.adapter.weight_ctcr=0.3 \
  model.adapter.threshold_max=0.9 model.adapter.seed=0
```

Streams: `source_only_{acdc4,csc12,acdcLT,cscLT}.py` (the name is historical: they only define the data and metrics). Smoke tests (80 / 100 frames): `smoke_ctcmt.py`, `smoke_csc.py`.
Development streams (ACDC val): `dev_acdcval.py`, `dev_acdcval_x5.py`, `dev_acdcval_x10.py`.

The experiments of the paper-style table were run with `docker/batch_mtl.sh` (one queue per GPU, nothing starts on its own):

```bash
export DATA_HOST=... CKPT=... OUT_ROOT=...
LR=3e-5 MAXGN=1000 bash docker/batch_mtl.sh plan             # prints the queues
bash docker/batch_mtl.sh dev2 0   &  bash docker/batch_mtl.sh dev2 1   &     # lr / clip check on ACDC val x10 (~1 h per run)
LR=3e-5 MAXGN=1000 bash docker/batch_mtl.sh final 0 &                        # GPU0: CTCMT on the four streams (~12 h)
LR=3e-5 MAXGN=1000 bash docker/batch_mtl.sh final 1 &                        # GPU1: Base
python tools/summarize_mtl_final.py $OUT_ROOT/final/*clip1000*              # per-stream and per-round table
```

Moraiti et al. single-task method: `docker/run_leviathan_eval.sh GPU configs/continuous/mean_teacher_adapter_yolox/moraiti_acdc4.py CKPT NAME` with `IMAGE=shift-detection-tta:cu113-mt`
(the Cityscapes / ACDC jsons must be in the class order of that checkpoint: `tools/data/reorder_coco_categories.py`).

## Results

`results/yolox_mtl_final/summary.md` (our method, all runs including the failed ones), `results/yolox_moraiti/summary.md` (single-task method on its own checkpoint).
Seed 0, one run per cell. See the top-level README for the table and the caveats.

`results/make_summary.py` regenerates `summary.md` and `per_domain.csv` from the raw run directories (logs are stored gzip-compressed under `logs/`).
