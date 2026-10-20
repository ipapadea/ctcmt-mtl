# YOLOX-MTL: what we built and how

Goal: run the final CTCMT method (jointly adapting detection and semantic segmentation) on a **different source architecture**,
a YOLOX-X detector with a semantic head, inside the `shift-detection-tta` codebase (mmdet 3 / mmengine, Moraiti et al.), and compare it with the
single-task mean-teacher method of that codebase. The Panoptic-FPN method it re-implements is described in [canonical_ctcmt.md](canonical_ctcmt.md).

## Components (all under `yolox-mtl/overlay/shift-detection-tta/`)

| What | File |
|---|---|
| YOLOX-X + semantic head on the shared PAFPN | `shift_tta/models/detectors/yolox_mtl_detector.py` (`YOLOXMTLDetector`), `shift_tta/models/heads/yolox_semseg_head.py` |
| Source training configs (v1 and v2) | `configs/source/yolox_mtl/yolox_x_mtl_{,v2_}8xb4-24e_cityscapes.py` (+ `amp_*`) |
| Final-method adapter | `shift_tta/models/adapters/mean_teacher_yolox_mtl_ctcmt_final_adapter.py` (`MeanTeacherYOLOXMTLAdapterCTCMTFinal`) |
| Older adapter (MSE consistency, 5 views; predates the final method, kept for reference) | `shift_tta/models/adapters/mean_teacher_yolox_mtl_ctcmt_adapter.py` |
| Per-domain metrics | `shift_tta/evaluation/metrics/coco_metric_domains.py`, `iou_metric_domains.py`, `iou_metric.py` |
| Dataset with the mask-path fix | `shift_tta/datasets/suffix_seg_dataset.py`, `cityscapes_seg_dataset.py` |
| Stream / experiment configs | `configs/continuous/mtl_final/`, `configs/continuous/no_adap_yolox/`, `configs/continuous/mean_teacher_adapter_yolox/moraiti_*.py` |
| Batch runner and docker helpers | `docker/batch_mtl.sh`, `docker/run_mtl.sh`, `docker/run_leviathan_eval.sh`, `docker/Dockerfile.leviathan{,.mt}` |
| Summaries | `tools/summarize_mtl_final.py`, `tools/summarize_yolox_moraiti.py`, `results/make_summary.py` |

## Source model

`YOLOXMTLDetector` = mmdet YOLOX-X (backbone, PAFPN, head) + `YOLOXSemSegHead` on the three PAFPN levels (3 convs per level, 256 hidden channels, `cat` fusion,
dropout 0.1, 19 classes). Trained for 24 epochs on Cityscapes, initialised from the COCO YOLOX-X.

v1 reached AP50 0.632 but a useless semantic head (mIoU 0.039). v2 (`yolox_x_mtl_v2_8xb4-24e_cityscapes.py`) changed several things at once:
no Mosaic / MixUp (they alpha-blend two images but keep one image's dense labels), class-weighted CE (median-frequency weights),
logits upsampled to the input resolution for the loss, a larger head, and semantic loss weight 2.0. It reaches AP50 0.731 / mAP 0.500 and mIoU 0.676 on Cityscapes val.
These changes were not ablated individually, so we do not know which one fixed the head. The final experiments use v2 `epoch_24.pth`.

## The adapter: what is the same as `CTCMT_MTL`

One frame at a time, online, no reset: weak-view EMA teacher -> pseudo-boxes (score >= 0.05, NMS 0.5) -> dynamic per-class thresholds
(init 0.8, clip [0.7, 0.9] ACDC / [0.7, 0.8] Cityscapes-C, `0.95 tau + 0.05 * 1.3 sqrt(s)`) -> score-EMA gate (EMA 0.7, ratio outside [1/1.4, 1.4] closes `L_det` only);
student on a strong view; `L = L_det + L_seg + 0.1 L_anchor + 0.3 L_CTCR`;
soft CE with class balancing (marginal EMA 0.999, `m^-0.5`) and loss-scale preservation; anchor marginal KL; conditional teacher aug-averaging (anchor confidence < 0.95,
scales {0.75, 1, 1.25} x flips); CT-CR with weight `0.2 + 0.8 q_T`, ignore outside boxes, global normalisation, not gated; one SGD step; EMA 0.9998;
Bernoulli restore p = 0.01 (0.001 on backbone and neck); prediction by the EMA teacher on the weak view after the update.
`adapt_enabled=False` gives the Source row through the very same prediction path.

## The adapter: deliberate differences

1. **Detection loss.** YOLOX head loss (`loss_cls`, `loss_bbox`, `loss_obj`, SimOTA assignment) on the hard pseudo-boxes, instead of RPN + ROI losses.
2. **Strong view.** SimCLR-style colour jitter (0.4, 0.4, 0.4, 0.1; p 0.8), grayscale (p 0.2), Gaussian blur (p 0.5, sigma 0.1-2) and three random erasings,
   computed on the GPU on the resized and padded tensor (no geometric change, so pseudo-boxes stay valid). The Panoptic-FPN code computes the same recipe with PIL in the dataset mapper;
   the blur kernel here is derived from sigma and is not bit-identical to PIL's.
3. **BatchNorm** layers stay in eval mode (frozen running statistics), the counterpart of the FrozenBN of the Panoptic-FPN source. The BN buffers are still EMA-ed into the teacher.
4. **Optimiser.** SGD with momentum 0.9, weight decay 1e-4 on weights only, **lr 3e-5** (instead of 1e-3).
5. **Gradient-norm clipping at 1000 and a non-finite guard** (`max_grad_norm`; steps with a non-finite loss or gradient are skipped, the EMA and restore still run).
   Neither exists in `CTCMT_MTL`. Why they exist: see below.
6. **Class order.** mmdet derives the detection labels from the **order of `categories` in the COCO json, not from `metainfo.classes`**. The Cityscapes / ACDC jsons list
   person, rider, car, truck, bus, train, motorcycle, bicycle, which is the order the source model was trained with and the order assumed by the class mapping of CT-CR (trainIds 11-18).
   Evaluating with another json order silently scores some classes 0.
7. **Seed.** `tools/test.py` fixes `seed = 1234`; the adapter's `seed` option re-seeds torch inside the adapter so that runs with different seeds differ.

Only the final-method components are in the adapter: no CT-CL, no CTPV, no gradient routing, no Fisher variants.

## Evaluation protocol

Same domain streams as the Panoptic-FPN experiments: ACDC4 = fog, night, rain, snow (train split, 400 images each);
CSC12 = 12 Cityscapes-C corruptions at severity 5 (500 images each); ACDC-LT = ACDC4 x10 (40 domains); CSC-LT = fog, motion_blur, snow, brightness, defocus_blur x10 (50 domains).
Detection (COCO AP50 / mAP) and mIoU are computed **per domain** and averaged over the domains (`CocoMetricDomains`, `IoUMetricDomains`).
Hyper-parameters were chosen on a **development stream** that is disjoint from the benchmark streams: the ACDC **val** split (fog 100, night 106, rain 100, snow 100),
alone or repeated 5 or 10 times. Cityscapes-C has no development stream (the noise corruptions are not available to us), so the CSC streams were never used for selection.

## How the numbers were obtained (chronological)

1. Source-only evaluation on ACDC4 / CSC12 with the per-domain metrics.
2. First adaptation smoke tests with the Panoptic-FPN values (lr 1e-3): the YOLOX objectness loss makes the first update explode (`loss_obj` ~ 800); lr 1e-4 diverges to NaN within ~20 frames.
3. Development sweep on ACDC val (single pass): lr 1e-6 ... 1e-4. 3e-5 is the best stable value (AP50 46.1 vs 44.3 Source); 1e-4 diverges. A run with lr = 0 reproduces the Source numbers (AP50 44.23 vs 44.25, mIoU identical), which validates the prediction path.
4. First batch of final runs with lr 3e-5 and **no** clipping (`summary.md`, "First batch"): Base finishes everywhere, but CTCMT diverges to NaN on CSC12 (by frame 50) and on CSC-LT (around frame 13700).
   The evaluation then crashed on the empty detections (fixed in `CocoMetricDomains`).
5. Gradient spikes of the objectness loss (norm ~4e4 vs ~3e2 typical) were the cause in the smoke test that reproduced the NaN (CSC12, frame 26). We added gradient-norm clipping and the non-finite guard,
   and picked the clip value (1000 vs 300) and the lr (3e-5 vs 1e-5) on the ACDC-val x10 stream. lr 3e-5 / clip 1000 was best (AP50 50.05, mIoU 47.60, stable over 10 rounds).
6. Second batch of final runs (lr 3e-5, clip 1000, seed 0) for CTCMT and Base on the four streams: no NaN, no skipped step. These are the numbers in the README.
   Note that clipping also changed runs that had not diverged: Base on CSC12 improved from AP50 23.01 (no clip) to 28.34.

## What the results do and do not show

- Adaptation helps in the short-term streams (AP50 +4.5 to +5.2 on ACDC4 and +3.1 to +4.4 on CSC12 over Source, depending on Base / CTCMT), and the mIoU stays above Source in every round of the long-term streams.
- **CT-CR (CTCMT vs Base) does not help on this architecture** at the weight and learning rate we used: CTCMT is below Base on all four streams (-0.7, -1.2, -1.3, -0.5 AP50). One seed, so differences of this size are not conclusive,
  but there is no sign of a gain, unlike with Panoptic-FPN (+1.2 AP50 on ACDC4 over 3 seeds).
- **Detection collapses in the long-term streams** for both Base and CTCMT: the per-round AP50 peaks in round 2 and falls below Source from round 7 (CTCMT) or 8 (Base) on ACDC-LT and from round 8 on CSC-LT, while mIoU stays above Source.
  The dev stream x10 did not predict this (it kept improving). We have not diagnosed the cause; in particular we cannot tell whether it comes from the adapter, from the method on this architecture or from the ACDC streams having 4x more frames per domain than the dev stream.
- Everything is seed 0 only; there are no error bars.
- The adapter is a re-implementation checked only by the lr = 0 test (prediction path) and by the loss traces; it is not verified line by line against `CTCMT_MTL`.

## Comparison with a single-task method

`configs/continuous/mean_teacher_adapter_yolox/moraiti_*.py` run the official `MeanTeacherYOLOXAdapterContrastive` (mean teacher + contrastive learning + stochastic restoration, detection only)
of Moraiti et al. with the hyper-parameters of its committed Cityscapes config (1 optimisation step, EMA 0.99965, consistency weight 0.005, contrastive weight 0.003, restoration p 0.01, 2 views, SGD lr 2.5e-4) on **its own** YOLOX-X Cityscapes checkpoint
(not redistributed here; md5 `f79e0f1f6cbc56513756ddd1d1dfaeb0`). This checkpoint is a different source model from our MTL one (Source AP50 48.2 vs 45.4 on ACDC4), so the two sets of numbers must not be compared as if they were the same experiment.
Notes for reproducing it: the mmdet class-order rule above applies (the AMROD jsons have to be re-ordered with `tools/data/reorder_coco_categories.py`);
`CocoMetric` caches the image ids / category ids / COCO api and has to be reset between domains (done in `CocoMetricDomains`); the adapter needs `albumentations==1.0.0` and `mmpretrain==1.0.0rc8` (see `docker/Dockerfile.leviathan.mt`).
