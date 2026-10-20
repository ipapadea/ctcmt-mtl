#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL
from detectron2.structures import Boxes, Instances

torch.manual_seed(7)

m = CTCMT_MTL.__new__(CTCMT_MTL)
nn.Module.__init__(m)
m.ctcr_mode = "per_box_full"
m.ctcr_mask_thresh = 0.3
m.ctcr_weight_floor = 0.2
m.ctcr_score_order = False
m.ctcr_det_score_weight = False
m.ctcr_det_score_gamma = 1.0

inst = Instances((8, 8))
inst.pred_boxes = Boxes(torch.tensor([
    [0.0, 0.0, 8.0, 8.0],
    [0.0, 0.0, 4.0, 4.0],
]))
inst.pred_classes = torch.tensor([2, 0], dtype=torch.long)
inst.scores = torch.tensor([0.99, 0.99])

logits = torch.randn(1, 19, 8, 8, requires_grad=True)

loss = m._ctcr_loss(logits, [inst], teacher_seg_probs=None)
assert loss is not None

target_car = torch.full((1, 8, 8), 13, dtype=torch.long)
target_person = torch.full((1, 4, 4), 11, dtype=torch.long)

l_car = F.cross_entropy(logits[:, :, 0:8, 0:8], target_car, reduction="none")[0].mean()
l_person = F.cross_entropy(logits[:, :, 0:4, 0:4], target_person, reduction="none")[0].mean()
expected = (l_car + l_person) / 2.0

assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-6), (loss.item(), expected.item())

fake_probs = torch.softmax(torch.randn(1, 19, 8, 8), dim=1)
loss_with_probs = m._ctcr_loss(logits, [inst], teacher_seg_probs=fake_probs)
assert torch.allclose(loss, loss_with_probs, atol=0.0, rtol=0.0)

loss.backward()
assert logits.grad is not None
assert torch.isfinite(logits.grad).all()

print("A2 UNIT TEST PASSED")
print(f"loss={loss.item():.8f}")
print("Verified: full bbox, per-box normalization, equal box averaging, no semantic weighting.")
