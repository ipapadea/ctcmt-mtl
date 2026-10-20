# CTCMT: continual test-time adaptation of joint object detection and semantic segmentation

CTCMT adapts a multi-task network (object detection + semantic segmentation sharing a backbone) to a stream of changing domains at test time,
frame by frame, without labels and without resetting. A mean teacher produces (i) hard detection pseudo-labels with dynamic per-class thresholds behind a score-EMA gate,
and (ii) soft semantic targets; a cross-task consistency regulariser (CT-CR) uses the detected boxes to supervise the semantic head, weighted by how much the semantic teacher agrees;
a frozen source copy anchors the class marginals and the weights are stochastically restored towards the source.

This repository contains the code of the experiments on two source architectures and a precise description of what the code does.

| Directory | Content |
|---|---|
| [`amrod/`](amrod/README.md) | **Panoptic-FPN R50 (Detectron2).** The reference implementation `CTCMT_MTL`, baselines on the same source model, the port of a single-task method (MTCLSR), run scripts and results |
| [`yolox-mtl/`](yolox-mtl/README.md) | **YOLOX-X + semantic head (mmdet).** Re-implementation of the final method, per-domain metrics, batch runner and results, plus the single-task method of Moraiti et al. on its own checkpoint |
| [`docs/canonical_ctcmt.md`](docs/canonical_ctcmt.md) | The method **as the code implements it**: every loss, gate, threshold and hyper-parameter, with the function that implements it |
| [`docs/yolox_mtl.md`](docs/yolox_mtl.md) | How the YOLOX-MTL version was built, how it differs from the Panoptic-FPN one, and a chronological account of the decisions and problems |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | Upstream repositories, pinned commits and licenses |

Both parts are **overlays**: `setup.sh` in each directory clones the upstream repository at a pinned commit, applies our patch and copies our files.

```bash
bash amrod/setup.sh ~/amrod_ctcmt          # then follow amrod/README.md
bash yolox-mtl/setup.sh ~/yolox_mtl        # then follow yolox-mtl/README.md
```

## The final objective

```
L = L_det + L_seg + 0.1 * L_anchor + 0.3 * L_CTCR
```

- `L_det`: standard detector losses on **hard** pseudo-labels (teacher, weak view); per-class thresholds `tau_c <- clip(0.95 tau_c + 0.05 * 1.3 sqrt(mean score of c))`; active only when the score-EMA gate is open.
- `L_seg`: soft cross-entropy to the teacher softmax (aug-averaged when the frozen anchor is unsure), class-balanced, loss-scale preserving.
- `L_anchor`: KL between the class marginals of the frozen source model and the student.
- `L_CTCR`: pixels in a pseudo-box get the box class as hard target, weighted by `0.2 + 0.8 q_T(class)`, ignored outside boxes, not gated.
- Student on a strong view, EMA teacher (0.9998) on the weak view, restore p = 0.01 (0.001 on the shared backbone), prediction by the EMA teacher after the update.

CT-CL (cross-task contrastive loss), CTPV (cross-task pseudo-label verification), gradient routing and Fisher-based restoration were explored; they are implemented behind flags and are **not** part of the final method.
Details in [docs/canonical_ctcmt.md](docs/canonical_ctcmt.md).

## Results

All numbers: mean over the domains of the stream, AP50 / mIoU in %. Streams: ACDC4 (fog, night, rain, snow), CSC12 (12 Cityscapes-C corruptions), and the long-term versions ACDC-LT (ACDC4 x10) and CSC-LT (5 corruptions x10).

### Panoptic-FPN R50 (`amrod/`), mean ± std over 3 seeds

| Stream | Source (AP50) | Base | CTCMT |
|---|---|---|---|
| ACDC4 | 34.08 | 36.84 ± 0.07 / 36.86 ± 0.17 | **38.01 ± 0.25 / 37.80 ± 0.06** |
| CSC12 | 13.46 | 16.96 ± 0.29 / 28.95 ± 0.67 | **17.72 ± 0.26 / 30.07 ± 0.19** |

Base = the same pipeline without CT-CR. Predicting before instead of after the update changes these numbers by at most 0.15 ([summary](amrod/results/preupdate_prediction/summary.md)).
The long-term streams and the other baselines are produced by the scripts in `amrod/scripts/`; their result tables are not part of this release.

### YOLOX-X + semantic head (`yolox-mtl/`), one seed

| Stream | Source | Base | CTCMT |
|---|---|---|---|
| ACDC4 | 45.40 / 43.75 | **50.60** / 46.09 | 49.88 / 46.05 |
| CSC12 | 23.97 / 33.77 | **28.34** / 39.85 | 27.10 / 38.30 |
| ACDC-LT | 45.40 / 43.75 | **47.73** / 49.58 | 46.46 / 48.20 |
| CSC-LT | 28.30 / 38.62 | **31.60** / 47.02 | 31.07 / 46.00 |

(Source on the long-term streams is its short-term per-domain scores, repeated.) Lr 3e-5, gradient-norm clip 1000, selected on a development stream disjoint from the benchmark.

### Single-task baseline on its own YOLOX-X checkpoint (`yolox-mtl/results/yolox_moraiti`), AP50, one seed

| Stream | Source | MT + CL + SR (Moraiti et al.) |
|---|---|---|
| ACDC4 | 48.20 | 51.72 |
| CSC12 | 44.26 | 46.33 |
| ACDC-LT | 48.20 | 52.51 |
| CSC-LT | 45.42 | 50.32 |

This is a **different source model** (higher Source AP50) from the MTL one above; the two tables are not the same experiment.

## What the results do not support (read before citing)

- **On YOLOX, CT-CR does not help**: CTCMT is below Base on all four streams, by 0.5-1.3 AP50. One seed, so small differences are inconclusive, but there is no sign of the +1.2 AP50 seen on Panoptic-FPN (ACDC4, 3 seeds).
- **On YOLOX, detection AP50 collapses in the long-term streams** (peak in round 2, below Source from round 7-8), for both Base and CTCMT; the mIoU stays above Source. We did not diagnose the cause. The per-round curves are in [`yolox-mtl/results/yolox_mtl_final/summary.md`](yolox-mtl/results/yolox_mtl_final/summary.md).
- **YOLOX needed two additions to `CTCMT_MTL`**: a much smaller learning rate (3e-5 instead of 1e-3) and gradient-norm clipping with a non-finite guard. Without clipping CTCMT diverged to NaN on Cityscapes-C (first-batch results are kept in the summary).
- **The YOLOX adapter is a re-implementation**, checked by an `lr = 0` run that reproduces Source and by loss traces, not line by line against `CTCMT_MTL`.
- **MTCLSR** (our port of the single-task method of Moraiti et al. to Panoptic-FPN) is below Source in all four protocols and degrades over the long-term rounds ([summary](amrod/results/mtclsr/summary.md)).
  Whether this is a bug of the port, an architecture effect or a hyper-parameter mismatch is **unknown**; do not read it as a result about the original method (which works on its own YOLOX checkpoint, table above).
- Restoration has no measured benefit on Panoptic-FPN: a seed-0 run without it is not worse than the headline run (ACDC4 AP50 39.40 vs 38.29, CSC12 18.09 vs 18.03). It is kept as a design component.
- Seeds: Panoptic-FPN short-term runs use 3 seeds; everything else is seed 0.

## Why and how (summary; details in the docs)

- **Code is the ground truth.** The description in `docs/canonical_ctcmt.md` was produced by auditing the code line by line against our slides; every mismatch was resolved in favour of the code.
- **Adapt-then-predict.** CTCMT predicts with the EMA teacher after the update of the frame. AMROD, CTAOD, OBAO and DAT-BF predict before it, so we added `SOLVER.CTCMT_PREDICT_BEFORE_UPDATE` and re-ran CTCMT and Base with it (3 seeds, two protocols): the difference is within noise.
- **Overlay, not fork.** The upstream AMROD repository has no license; we ship patches and new files and a script that rebuilds the tree from the pinned upstream commit (verified: the rebuilt tree is byte-identical to our working tree for all shipped files, except for the removal of machine-specific hosts and paths).
- **A second architecture** was added to check that the method is not specific to Faster R-CNN. It required a new source model (v2 recipe), per-domain metrics for the mmdet code base, and the changes listed in `docs/yolox_mtl.md`.
- **A development stream** (ACDC val) disjoint from the benchmark streams was used to choose the learning rate and clip. Cityscapes-C has no held-out corruptions, so its streams were never used for selection.
- **Failures are kept**: diverged runs, the first batch without clipping and the MTCLSR port are in the results, not hidden.

## License

Third-party components keep their own licenses (see [THIRD_PARTY.md](THIRD_PARTY.md)). Our own files are MIT-licensed (`LICENSE`); this does not extend to the upstream code, to the patches' context lines, or to the ports listed in THIRD_PARTY.md.
