# YOLOX-X: Source-only vs MT + CL + SR (Moraiti et al.)

Checkpoint: `yolox_x_8xb4-24e_cityscapes.pth` of the authors (md5 `f79e0f1f6cbc56513756ddd1d1dfaeb0`). Seed 0 (config `seed=1234`), one run per protocol. Official adapter `MeanTeacherYOLOXAdapterContrastive` with the hyper-parameters of the committed Cityscapes config (optim_steps=1, EMA 0.99965, consistency 0.005, contrastive 0.003, restoration p=0.01, 2 views, SGD lr 0.00025). Mean over the domains of the stream.

| Protocol | Source AP50 | Moraiti AP50 | Δ AP50 | Source mAP | Moraiti mAP | Δ mAP |
|---|---|---|---|---|---|---|
| ACDC4 | 48.20 | 51.72 | +3.52 | 25.50 | 27.67 | +2.17 |
| CSC12 | 44.26 | 46.33 | +2.07 | 26.10 | 28.00 | +1.90 |
| ACDC-LT | 48.20 | 52.51 | +4.30 | 25.50 | 29.00 | +3.50 |
| CSC-LT | 45.42 | 50.32 | +4.90 | 27.18 | 30.95 | +3.77 |

## Long-term: mean AP50 per round (Moraiti)

- ACDC-LT (4 domains/round): 51.7, 52.6, 52.6, 52.9, 52.7, 52.6, 52.5, 52.6, 52.5, 52.6
- CSC-LT (5 domains/round): 49.9, 50.6, 50.5, 50.4, 50.4, 50.3, 50.3, 50.3, 50.3, 50.3

## CSC12 per domain (AP50)

| Domain | Source | Moraiti | Δ |
|---|---|---|---|
| defocus_blur | 44.4 | 44.7 | +0.3 |
| glass_blur | 39.7 | 40.5 | +0.8 |
| motion_blur | 39.7 | 40.7 | +1.0 |
| zoom_blur | 14.9 | 15.2 | +0.3 |
| snow | 14.6 | 36.4 | +21.8 |
| frost | 45.8 | 49.7 | +3.9 |
| fog | 66.8 | 66.2 | -0.6 |
| brightness | 61.6 | 65.4 | +3.8 |
| contrast | 49.4 | 31.6 | -17.8 |
| elastic_transform | 63.4 | 65.5 | +2.1 |
| pixelate | 56.9 | 59.4 | +2.5 |
| jpeg_compression | 33.9 | 40.6 | +6.7 |

## ACDC4 per domain (AP50)

| Domain | Source | Moraiti | Δ |
|---|---|---|---|
| fog | 67.1 | 68.8 | +1.7 |
| night | 33.3 | 35.3 | +2.0 |
| rain | 45.6 | 48.8 | +3.2 |
| snow | 46.8 | 54.0 | +7.2 |

## Caveats

- Cross-architecture: YOLOX-X has its own, much higher, Source than the Panoptic-FPN R50 of the CTCMT tables; only Source vs MT+CL+SR on the same YOLOX is a like-for-like comparison.
- Detection only, one seed, no error bars.
- optim_steps=1 as in the committed config; the paper text says 5.
- Class order: mmdet takes labels from the *order of `categories` in the json*, so the jsons were re-ordered to the checkpoint order (`tools/data/reorder_coco_categories.py`). Without that five classes score 0.
- Source on clean Cityscapes val is 42.2 mAP here vs 40.3 in the paper (Table 5); cause not determined.
- The long-term Source rows are the short-term domain results repeated, not separate runs.
