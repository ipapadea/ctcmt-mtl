# CT-CMT MTL extension of shift-detection-tta — setup & run guide

This extension adds a **multi-task YOLOX** (detection + semantic segmentation)
and the **Cross-Task Contrastive Mean Teacher (CT-CMT)** continual TTA method
on top of the published single-task infrastructure. Prior single-task code
files are untouched; everything below lives in **new** files.

## Files added

```
shift_tta/models/heads/
  ├─ __init__.py                       (new)
  └─ yolox_semseg_head.py              (new)
shift_tta/models/detectors/
  └─ yolox_mtl_detector.py             (new)
shift_tta/models/losses/
  ├─ mtl_consistency_loss.py           (new)
  └─ cross_task_contrastive_loss.py    (new)
shift_tta/models/adapters/
  └─ mean_teacher_yolox_mtl_ctcmt_adapter.py   (new)
configs/source/yolox_mtl/
  └─ yolox_x_mtl_8xb4-24e_cityscapes.py         (new)
configs/continuous/ctcmt/
  ├─ ctcmt_yolox_mtl_cityscapes_to_acdc.py     (new)
  └─ ctcmt_yolox_mtl_cityscapes_to_foggy.py    (new)
```

`__init__.py` in `losses/`, `detectors/`, `adapters/` was extended to export
the new classes. No other files were modified.

## Environment requirements

Same as the published `shift-detection-tta`:

- Python 3.8+
- PyTorch (whatever matches your mmcv build, usually 1.11–1.13)
- **mmengine, mmdet 3.x, mmtrack** (installed per the original `docs/get_started.md`)
- `torchvision` (needed by `roi_align` in `cross_task_contrastive_loss.py`)

No new pip dependencies.

## Dataset preparation

### Cityscapes (source)

The source training config expects:

```
data/cityscapes/
  ├─ leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
  ├─ leftImg8bit/{train,val}/det_2d_cocoformat.json   ← the COCO detection json
  └─ gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png
```

The `labelTrainIds.png` masks encode the 19 Cityscapes trainIds (0..18 + 255).
Generate them (once) from `labelIds.png` via
[`cityscapesscripts.preparation.createTrainIdLabelImgs`](https://github.com/mcordts/cityscapesScripts).

If you already have the standard `labelIds.png` files (which is what the
Cityscapes zip ships), run:

```bash
cd /path/to/cityscapesScripts
python -m cityscapesscripts.preparation.createTrainIdLabelImgs
```

pointing `CITYSCAPES_DATASET` at your Cityscapes root.

### ACDC (target)

Config expects:

```
data/acdc/
  ├─ rgb_anon/<weather>/<split>/<seq>/*_rgb_anon.png
  ├─ gt/<weather>/<split>/<seq>/*_gt_labelTrainIds.png
  └─ gt_detection/<weather>/instancesonly_<weather>_{train,val}_gt_detection.json
```

which matches the official 2023 ACDC release exactly (RGB + labelTrainIds
+ instancesonly detection jsons).

### Foggy Cityscapes (secondary target)

```
data/cityscapes_foggy/
  └─ leftImg8bit_foggyDBF/val/<city>/*_leftImg8bit_foggy_beta_{0.005,0.01,0.02}.png
```

The Foggy Cityscapes val config reuses the same detection json and gtFine
semseg masks as clear Cityscapes val (labels don't change with fog density).

## Running

### 1. Source training

```bash
tools/train.py configs/source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py
```

Same 24-epoch schedule and hyperparams as your published single-task config.
Expected outputs: `loss_bbox`, `loss_obj`, `loss_cls`, `loss_seg` per iteration.
Validation runs both `mmdet.CocoMetric` (bbox) and `mmdet.IoUMetric` (mIoU).

Expected metric magnitudes on clear Cityscapes val (from the source config):
- **bbox mAP** ≈ your prior single-task YOLOX-X on Cityscapes
- **seg mIoU** ≈ 68–75% (from-scratch seg head on top of COCO-pretrained YOLOX
  backbone; Panoptic FPN R-50 gets ~74% in the same setup).

If `loss_seg` doesn't decrease in the first ~500 iterations, verify that
`gt_sem_seg` is actually reaching the detector — inspect one training sample
via `tools/misc/browse_dataset.py`.

### 2. Continual TTA — Cityscapes → ACDC

```bash
tools/test.py configs/continuous/ctcmt/ctcmt_yolox_mtl_cityscapes_to_acdc.py \
    --checkpoint work_dirs/yolox_x_mtl_8xb4-24e_cityscapes/epoch_24.pth
```

Streams ACDC val in fog→night→rain→snow order and adapts the model per image.
Reports COCO-format mAP and mIoU at the end.

### 3. Continual TTA — Cityscapes → Foggy Cityscapes

```bash
tools/test.py configs/continuous/ctcmt/ctcmt_yolox_mtl_cityscapes_to_foggy.py \
    --checkpoint work_dirs/yolox_x_mtl_8xb4-24e_cityscapes/epoch_24.pth
```

Streams Foggy Cityscapes val at beta=0.005 → 0.01 → 0.02 → clear (loopback).
Matches the "Low Fog / Medium Fog / High Fog / Clear" columns of your prior
paper's Table 5.

## Ablations (recommended for the paper)

Toggle at the config level:

| Ablation | How |
|---|---|
| MT only (no CT-CL, no CT-CR) | `ct_cl_loss.weight = 0`, `ct_cr_weight = 0`, `mtl_consistency_loss.weight_seg = 0` (falls back close to your prior single-task) |
| MT + object-level CL (prior paper's setup, single-task) | `ct_cl_loss.include_seg_view = False`, `ct_cr_weight = 0`, `mtl_consistency_loss.weight_seg = 0` |
| MT + CT-CL (no CT-CR) | `ct_cr_weight = 0` |
| Full CT-CMT | default config |
| + Task-weighted SR | `task_weighted_restore = True` |

## Known limitations / things to sanity-check

1. **The CT-CR bbox-to-seg pixel-space mapping** in
   `MeanTeacherYOLOXMTLAdapterCTCMT._ct_cr_loss` uses an approximate scaling
   (`sx = W_logits / max_x`). If your input aspect ratio differs strongly from
   the seg-logits grid, replace with proper metainfo-driven scaling. It's
   only used for a small-weight regularizer so the approximation is safe.

2. **Teacher module construction** uses `mmtrack.ExpMomentumEMA` with
   `momentum = 1 - alpha`. The alpha in your prior paper is 0.99965 for
   Cityscapes and 0.995 for COCO-C — adjust in the CTTA config if targeting
   COCO-C.

3. **Seg-head init** is random (no pretrained weights available). The COCO
   pretrained YOLOX backbone provides a good starting point for the shared
   backbone, but the seg head needs the 24 epochs of source training to
   converge. Expect the first few epochs to have very high `loss_seg`.

4. **`YOLOXSemSegHead.in_channels`** must match your YOLOX size:
   - `yolox_x`: (320, 320, 320)
   - `yolox_s`: (128, 128, 128)
   - `yolox_l`: (256, 256, 256)

5. **Loss weights** in the CTTA configs are chosen conservatively:
   `weight_det=0.005`, `weight_seg=1.0`, `weight_ct_cl=0.003`, `ct_cr_weight=0.05`.
   Tune per benchmark if seg loss dominates the encoder gradient.
