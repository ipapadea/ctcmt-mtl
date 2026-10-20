# CTCMT as implemented: `CTCMT_MTL` (Panoptic-FPN)

This page describes what the code in `amrod/overlay/detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py` does for the
headline configuration (`detectron2/configs/Cityscapes/newsrc_{acdc,csc}_ctcr.yaml`). It was written by reading the
executable code, not the paper. Where a comment in the code disagrees with the code, the code is described.
All names below are functions / methods of `CTCMT_MTL` unless stated otherwise.

## Setting

One student, one EMA teacher and one frozen source copy (the "anchor") of a Panoptic-FPN R50 trained on Cityscapes.
Test stream with batch size 1, online, never reset. Detection pseudo-labels and the semantic soft targets both come from the
teacher; only the student is trained.

## 1. Algorithm flow (`forward`)

Per frame:

1. `_teacher_pseudo`: teacher inference on the **weak** view, no gradient. Dynamic per-class thresholds and the score-EMA gate (section 7).
2. CTPV filter (`_ctpv_filter`), only if enabled. **Off** in CTCMT.
3. Student forward on the **strong** view (`image_strong` from the dataset mapper: colour jitter, grayscale, blur, three random erasings).
4. Losses of sections 2-4 (CT-CL of section 5 is off), one backward pass, one SGD step.
5. `_update_teacher` (EMA), then `_stochastic_restore` (student towards the source).
6. The returned prediction is a second `teacher.inference` on the **weak** view, **after** the update (adapt-then-predict).
   The pseudo-labels of the frame come from the pre-update teacher.

`SOLVER.CTCMT_PREDICT_BEFORE_UPDATE=True` switches step 6 to predict-then-adapt (the teacher output is frozen at the start of the frame). The
pre/post difference is at most 0.15 AP50 / mIoU with varying sign (`amrod/results/preupdate_prediction/summary.md`).

## 2. Detection loss `L_det`

Branch "Det consistency" of `forward`. Applied only if the gate is open and there is at least one pseudo-box.
It is the standard `proposal_generator` + `roi_heads` loss of the student (`loss_rpn_cls`, `loss_rpn_loc`, `loss_cls`, `loss_box_reg`, weight 1.0)
on **hard pseudo ground truth** (`_to_gt_instances`): the teacher boxes and classes after NMS 0.5, score >= 0.05 and the per-class thresholds.
There is no soft consistency.

Dynamic thresholds (`_dyn_thresholds`, called from `_teacher_pseudo`):

- `s_c` = mean score of class `c` among detections with score > 0.1 (hard-coded floor).
- Class with such detections: `tau_c <- clip(0.95 * tau_c + 0.05 * 1.3 * sqrt(s_c), 0.7, tau_max)`.
- Class without detections: `tau_c` unchanged (only re-clipped). `tau_c` starts at 0.8.
- `tau_max` = 0.9 on ACDC and 0.8 on Cityscapes-C.
- The thresholds are updated also when the gate is closed, but not when the teacher has no detection with score > 0.1.

## 3. Semantic loss `L_seg`

Branch "Sem-seg soft-CE":

- `l_i = - sum_k q_T[i,k] log p_S[i,k]`: soft cross-entropy between the teacher softmax `q_T` and the student softmax `p_S`.
  It is not hard CE and not entropy minimisation.
- Class balancing (`CLASS_BALANCED_CE`): running EMA (0.999) of the teacher class marginal `m` (`_update_class_marginal`),
  weight `w_i = m[argmax q_T(i)]^-0.5`, normalised to mean 1. `L_cb = sum(w l) / (sum(w) + 1e-6)`.
- Loss-scale preservation (`SEG_LOSS_SCALE_PRESERVE`): `L_seg = kappa * L_cb`, `kappa = stopgrad(mean(l) / L_cb)`.
- Anchor term `seg/anchor_marginal`: `0.1 * KL(q_A || q_S) = 0.1 * sum_k q_A[k] (log q_A[k] - log q_S[k])`, where `q_A`, `q_S` are the
  image-level mean class marginals of the frozen anchor (no gradient) and of the student. The anchor sees the strong view.
  There is no pixel-wise anchor loss.
- Conditional augmentation averaging (`_teacher_aug_avg_seg_probs`): if the mean pixel-wise max probability of the **anchor** is < 0.95,
  `q_T` is the **teacher** softmax averaged over scales {0.75, 1.0, 1.25} x flips {off, on} on the student's strong image tensor; otherwise it is the single-pass weak-view teacher softmax.
  It fires on 97-99 % of the frames. The averaged `q_T` is used for the soft-CE target, the class marginal and weights and for CT-CR.
  It is not used for the detection pseudo-labels or the final prediction.

## 4. Cross-task consistency regulariser `L_CTCR` (`_ctcr_loss`, mode `soft_seg_global`)

- Detection class -> Cityscapes trainId: person, rider, car, truck, bus, train, motorcycle, bicycle -> 11, 12, ..., 18.
- Every pixel inside a pseudo-box gets the hard target `sigma(c)`; pixels outside all boxes are ignored (`ignore_index` 255, weight 0).
- Weight `w = 0.2 + 0.8 * q_T(p, sigma(c))`, `q_T` detached (so the semantic teacher only decides how much to trust the box, not the target).
- Overlaps: boxes are processed in detector order and the last one overwrites target and weight. The detector order is score-descending, so the
  lowest-score box usually wins.
- `L_CTCR = sum(w * CE) / (sum(w) + 1e-6)`: one global normalisation over all supervised pixels, not per box. Added with weight 0.3.
- Conditions: weight > 0, at least one pseudo-box, (`det_gate` or `seg_gate`). It is **not** blocked by the score-EMA gate.
- The pseudo-boxes are those after the dynamic thresholds (and after CTPV if enabled).

## 5. CT-CL (`_ctcl_loss`, `_supcon_loss`): implemented, **off** in the final method

- One FPN level (the last of `roi_heads.box_in_features`, p5). `z_det = L2(mean_{7x7} RoIAlign(F, box))`;
  `z_seg = L2( sum_{p in box} q_T(p, sigma(c)) F_p / sum q_T )` (an integer crop, no RoIAlign). Raw features, **no** learned projection heads.
- SupCon with temperature 0.07, label = detection class, all same-class pairs positive; the mean runs over anchors with at least one positive.
- Requires `det_gate and seg_gate`, so the score gate disables it. Ablation weight 0.5.

## 6. CTPV (`_ctpv_filter`): implemented, **off** in the final method

Hard filter: `r = mean over box pixels of [argmax_k q_T = sigma(c)]` on the weak single-pass teacher softmax; keep the box if `r >= 0.3`, else reject.
Unmapped classes are kept. No down-weighting.

## 7. Score-EMA gate (`_teacher_pseudo`)

- `m` = EMA of the mean score of detections with score > 0.1; `r = mean_score / m`.
- The gate closes iff `r > 1.4` or `1/r > 1.4`, i.e. a sharp change in either direction. `m <- 0.7 m + 0.3 mean_score`, also when the gate is closed.
- It is closed also when the teacher has no detection (or none above 0.1); `m` and the thresholds are not updated on those paths.
- Effect: it disables **only** `L_det` and CT-CL. Soft CE, the anchor term and CT-CR continue; the optimiser step, EMA and restore always run.

## 8. EMA (`_update_teacher`)

Every frame, `teacher = 0.9998 * teacher + 0.0002 * student` for all floating state (including BN buffers); non-float entries are copied.

## 9. Stochastic restore (`_stochastic_restore`)

Every frame after the EMA, for each trainable `weight` / `bias` of the student, a Bernoulli element mask with `p = 0.01` replaces entries by the frozen source value;
`p = 0.001` for keys starting with `backbone.`, `fpn.` or `proposal_generator.anchor_generator.`. There is **no** Fisher computation in this path
(the flag `CROSS_TASK_FISHER` and the name of `fisher_*` are historical). The teacher is not restored directly.

## 10. Objective and optimiser

```
L = L_det + L_seg + 0.1 * L_anchor + 0.3 * L_CTCR
```

SGD, lr 1e-3, momentum 0.9, weight decay 1e-4, batch size 1.

## 11. Evaluation

`tools/train_net.py --eval-only`: test loader with batch size 1, one persistent model across the whole stream of datasets, no reset. The prediction is the EMA
teacher on the weak view after the update. Detection and semantic outputs are evaluated separately (no panoptic merge). Ground truth is never used
during adaptation (`CTCMT_MASK_DIAG_GT_ROOT` only feeds a logging diagnostic). Long-term streams repeat the same list of domains 10 times.

## Explored but not part of the final method

All flag-gated in `ctcmt_mtl.py` and off in the headline configs: CT-CL, CTPV, per-task / directional gates, gradient routing of the shared trunk
(`CONFLICT_MODE`: `aux_head_only`, `resnet_route`, `task_coco`), cross-task Fisher and Fisher-guided restore, prototype anchor, entropy-weighted CE, det-veto.
`CONFLICT_MODE=resnet_route` with `AUX_TRUNK_LAMBDA=1.0` (used in the headline config) reduces to ordinary joint back-propagation; it only adds a second backward pass.
Base = the same configuration with `WEIGHT_CTCR=0` (it still has the strong-view student, class balancing, anchor KL, aug-averaging, gate and restore).

## Stale comments in the code

- "skip step when teacher confidence is stable" (two places): the code skips when the confidence ratio leaves `[1/1.4, 1.4]`, i.e. when it is **not** stable.
- "every teacher/anchor path keeps the weak one": the anchor and the augmented teacher passes consume the student's strong image tensor; only the pre-update teacher inference and the final prediction use the weak view.
