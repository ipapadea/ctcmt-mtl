#!/usr/bin/env python3
"""Unit test for the detector-dominant CT-CR options (D1 + D2).

Checks:
  1. Both flags off reproduce the current loss EXACTLY, in every CT-CR mode.
  2. CTCMT_CTCR_SCORE_ORDER makes the HIGHEST-scoring box win an overlap
     (today the lowest-scoring one does, because detections arrive
     score-descending and later boxes overwrite earlier ones).
  3. CTCMT_CTCR_DET_SCORE_WEIGHT scales each box by det_score ** gamma, and
     gamma = 0 collapses back to the unweighted loss.
  4. Gradients stay finite.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.ctcmt_mtl import (
    CTCMT_MTL,
    _DET_TO_SEG_CLASS_CITYSCAPES,
)
from detectron2.structures import Boxes, Instances

torch.manual_seed(11)


def make(mode, order=False, dsw=False, gamma=1.0, floor=0.2):
    m = CTCMT_MTL.__new__(CTCMT_MTL)
    nn.Module.__init__(m)
    m.ctcr_mode = mode
    m.ctcr_mask_thresh = 0.3
    m.ctcr_weight_floor = floor
    m.ctcr_score_order = order
    m.ctcr_det_score_weight = dsw
    m.ctcr_det_score_gamma = gamma
    return m


# Box 1 is nested inside box 0 and scores LOWER, as detectron2 orders them.
inst = Instances((8, 8))
inst.pred_boxes = Boxes(torch.tensor([
    [0.0, 0.0, 8.0, 8.0],
    [0.0, 0.0, 4.0, 4.0],
]))
inst.pred_classes = torch.tensor([2, 0], dtype=torch.long)   # car, person
inst.scores = torch.tensor([0.99, 0.72])

logits = torch.randn(1, 19, 8, 8, requires_grad=True)

# Crafted so BOTH boxes clear the hard_seg mask; with random probs the
# low-scoring box is masked out entirely and a one-box weighted mean is a no-op.
CAR = _DET_TO_SEG_CLASS_CITYSCAPES[2]
PERSON = _DET_TO_SEG_CLASS_CITYSCAPES[0]
probs = torch.full((1, 19, 8, 8), 0.01)
probs[0, CAR] = 0.9
probs[0, PERSON, 0:4, 0:4] = 0.9
probs = probs / probs.sum(dim=1, keepdim=True)

MODES = ["full_box", "soft_seg_global", "per_box_full", "hard_seg", "soft_seg"]

# --- 1. flags off are a no-op -------------------------------------------------
for mode in MODES:
    ref = make(mode)._ctcr_loss(logits, [inst], probs)
    off = make(mode, order=False, dsw=False)._ctcr_loss(logits, [inst], probs)
    assert torch.equal(ref, off), f"{mode}: default path changed"
print(f"1. OK  flags off are bit-identical in all {len(MODES)} modes")

# --- 2. score ordering flips who wins the overlap ------------------------------
car, person = CAR, PERSON


def target_of(order):
    m = make("full_box", order=order)
    tgt = torch.full((1, 8, 8), 255, dtype=torch.long)
    scores = inst.scores.tolist()
    idx = sorted(range(2), key=lambda i: scores[i]) if order else range(2)
    for j in idx:
        x1, y1, x2, y2 = inst.pred_boxes.tensor[j].tolist()
        sc = _DET_TO_SEG_CLASS_CITYSCAPES[int(inst.pred_classes[j])]
        tgt[0, int(y1):int(y2), int(x1):int(x2)] = sc
    return tgt


assert target_of(False)[0, 0, 0] == person, "default: low-scoring box should win"
assert target_of(True)[0, 0, 0] == car, "score order: high-scoring box should win"
loss_ord = make("soft_seg_global", order=True)._ctcr_loss(logits, [inst], probs)
loss_def = make("soft_seg_global")._ctcr_loss(logits, [inst], probs)
assert not torch.allclose(loss_ord, loss_def), "score order had no effect"
print(f"2. OK  overlap winner flips  (loss {loss_def.item():.4f} -> {loss_ord.item():.4f})")

# --- 3. detector-score weighting ----------------------------------------------
# gamma = 0 makes every weight 1.0, so the loss must fall back to unweighted.
for mode in ["per_box_full", "hard_seg", "soft_seg"]:
    base = make(mode)._ctcr_loss(logits, [inst], probs)
    g0 = make(mode, dsw=True, gamma=0.0)._ctcr_loss(logits, [inst], probs)
    assert torch.allclose(base, g0, atol=1e-6), f"{mode}: gamma=0 should be a no-op"
    g1 = make(mode, dsw=True, gamma=1.0)._ctcr_loss(logits, [inst], probs)
    assert not torch.allclose(base, g1), f"{mode}: det-score weighting had no effect"

# full_box has no weight map until det-score weighting turns one on.
fb = make("full_box")._ctcr_loss(logits, [inst], probs)
fb_w = make("full_box", dsw=True, gamma=1.0)._ctcr_loss(logits, [inst], probs)
assert not torch.allclose(fb, fb_w), "full_box: det-score weighting had no effect"

# Explicit reference for the weighted per-box mean.
m = make("per_box_full", dsw=True, gamma=2.0)
got = m._ctcr_loss(logits, [inst], probs)
losses, weights = [], []
for j in range(2):
    x1, y1, x2, y2 = [int(v) for v in inst.pred_boxes.tensor[j].tolist()]
    sc = _DET_TO_SEG_CLASS_CITYSCAPES[int(inst.pred_classes[j])]
    crop = logits[0:1, :, y1:y2, x1:x2].float()
    tgt = torch.full((1, y2 - y1, x2 - x1), sc, dtype=torch.long)
    losses.append(F.cross_entropy(crop, tgt, reduction="none")[0].mean())
    weights.append(float(inst.scores[j]) ** 2.0)
w = torch.tensor(weights)
want = (torch.stack(losses) * w).sum() / w.sum()
assert torch.allclose(got, want, atol=1e-6), (got.item(), want.item())
print(f"3. OK  det-score weighting matches reference  ({got.item():.6f})")

# --- 4. gradients -------------------------------------------------------------
for mode in MODES:
    logits.grad = None
    make(mode, order=True, dsw=True, gamma=2.0)._ctcr_loss(
        logits, [inst], probs
    ).backward()
    assert torch.isfinite(logits.grad).all(), f"{mode}: non-finite grad"
print("4. OK  gradients finite in all modes")
print("\nALL PASS")
