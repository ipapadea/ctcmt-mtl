#!/usr/bin/env python3
"""Unit test for CT-CR mode D (soft_seg_global).

Checks:
  1. floor = 1.0 reduces D exactly to A (full_box).
  2. D matches an explicit weighted-CE reference with A's overwrite order.
  3. D differs from C (soft_seg) whenever box sizes differ.
  4. Gradients are finite.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL
from detectron2.structures import Boxes, Instances

torch.manual_seed(7)


def make(mode, floor):
    m = CTCMT_MTL.__new__(CTCMT_MTL)
    nn.Module.__init__(m)
    m.ctcr_mode = mode
    m.ctcr_mask_thresh = 0.3
    m.ctcr_weight_floor = floor
    m.ctcr_score_order = False
    m.ctcr_det_score_weight = False
    m.ctcr_det_score_gamma = 1.0
    return m


inst = Instances((8, 8))
# Score-descending order, as detectron2 returns. Box 2 overwrites box 1 in A/D.
inst.pred_boxes = Boxes(torch.tensor([
    [0.0, 0.0, 8.0, 8.0],
    [0.0, 0.0, 4.0, 4.0],
]))
inst.pred_classes = torch.tensor([2, 0], dtype=torch.long)  # car(13), person(11)
inst.scores = torch.tensor([0.99, 0.90])

logits = torch.randn(1, 19, 8, 8, requires_grad=True)
probs = torch.softmax(torch.randn(1, 19, 8, 8), dim=1)

# --- 1. floor = 1.0  =>  D == A -------------------------------------------
loss_a = make("full_box", 0.2)._ctcr_loss(logits, [inst], probs)
loss_d1 = make("soft_seg_global", 1.0)._ctcr_loss(logits, [inst], probs)
assert torch.allclose(loss_a, loss_d1, atol=1e-6), (loss_a.item(), loss_d1.item())

# --- 2. explicit reference for floor = 0.2 ---------------------------------
FLOOR = 0.2
target = torch.full((1, 8, 8), 255, dtype=torch.long)
weight = torch.zeros(1, 8, 8)
target[0, 0:8, 0:8] = 13
weight[0, 0:8, 0:8] = FLOOR + (1 - FLOOR) * probs[0, 13, 0:8, 0:8].clamp(0, 1)
target[0, 0:4, 0:4] = 11
weight[0, 0:4, 0:4] = FLOOR + (1 - FLOOR) * probs[0, 11, 0:4, 0:4].clamp(0, 1)

ce = F.cross_entropy(logits, target, ignore_index=255, reduction="none")
expected = (weight * ce).sum() / weight.sum()

loss_d = make("soft_seg_global", FLOOR)._ctcr_loss(logits, [inst], probs)
assert torch.allclose(loss_d, expected, atol=1e-7, rtol=1e-6), (loss_d.item(), expected.item())

# --- 3. D != C (normalization granularity differs) -------------------------
loss_c = make("soft_seg", FLOOR)._ctcr_loss(logits, [inst], probs)
assert not torch.allclose(loss_d, loss_c, atol=1e-4), "D and C must differ on unequal box sizes"

# --- 4. gradients ----------------------------------------------------------
loss_d.backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all()

print("D UNIT TEST PASSED")
print(f"  A  (full_box)        = {loss_a.item():.8f}")
print(f"  D  (floor=1.0)       = {loss_d1.item():.8f}   (== A)")
print(f"  D  (floor=0.2)       = {loss_d.item():.8f}")
print(f"  C  (soft_seg, 0.2)   = {loss_c.item():.8f}")
print("Verified: global target map, A's overwrite order, soft weights, per-pixel normalization.")
