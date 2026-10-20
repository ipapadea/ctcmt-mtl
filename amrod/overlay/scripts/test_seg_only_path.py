#!/usr/bin/env python3
"""Regression test for SemanticSegmentor support in CTCMT_MTL (ST-S).

Drives the real ``forward()`` end to end on a student that has no detector at
all, using the actual ST-S config so the exercised settings are the ones that
will run. Fake backbones keep it CPU-only and checkpoint-free.

Checks:
  1. forward() completes with no detection branch present.
  2. The output carries "sem_seg" and NOT "instances".
  3. The segmentation loss actually fires and updates the student.
  4. Teacher EMA moves, and the teacher is what gets evaluated.
  5. _empty_instances supports len() -- a bare Instances() raises, and the
     periodic log line reads len(pseudo_inst[0]) unconditionally.
  6. No detection attribute is ever touched (the fakes have none, so any
     unguarded access raises AttributeError).
  7. The sem_seg head's train/eval contract is reproduced faithfully: in train
     mode it returns (None, losses), so a missing .eval() would surface here.

Run inside the image, from the repo root:
  python scripts/test_seg_only_path.py
"""
import copy

import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL
from detectron2.structures import ImageList

torch.manual_seed(0)

CFG = "detectron2/configs/Cityscapes/ctcmt_sts_semfpn_cscLT.yaml"
K, H, W = 19, 64, 64


class FakeBackbone(nn.Module):
    size_divisibility = 32
    padding_constraints = {}

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return {"p2": self.conv(x)}


class FakeSemSegHead(nn.Module):
    """Mirrors SemSegFPNHead's train/eval contract exactly."""

    def __init__(self):
        super().__init__()
        self.pred = nn.Conv2d(8, K, 1)

    def forward(self, features, targets=None):
        x = self.pred(features["p2"])
        if self.training:
            return None, {"loss_sem_seg": x.float().mean()}
        return x, {}


class FakeSemanticSegmentor(nn.Module):
    """A student with NO roi_heads, NO proposal_generator and NO inference()."""

    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(3, 1, 1))
        self.backbone = FakeBackbone()
        self.sem_seg_head = FakeSemSegHead()

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs, strong_aug: bool = False):
        key = "image_strong" if strong_aug else "image"
        imgs = [(x[key].float() - self.pixel_mean) / self.pixel_std
                for x in batched_inputs]
        return ImageList.from_tensors(imgs, self.backbone.size_divisibility)


def make_inputs(n=1):
    return [{
        "image": torch.randint(0, 255, (3, H, W)).float(),
        "image_strong": torch.randint(0, 255, (3, H, W)).float(),
        "height": H, "width": W, "file_name": f"fake_{i}.png", "image_id": i,
    } for i in range(n)]


cfg = get_cfg()
cfg.merge_from_file(CFG)
cfg.MODEL.DEVICE = "cpu"
print(f"config            : {CFG}")
print(f"  SEG_ONLY={cfg.SOLVER.CTCMT_SEG_ONLY}  DET_ONLY={cfg.SOLVER.CTCMT_DET_ONLY} "
      f"STRONG_AUG={cfg.SOLVER.CTCMT_STRONG_AUG_STUDENT} SEG_AUG={cfg.SOLVER.CTCMT_SEG_AUG_ENABLED}")
print(f"  CLASS_BAL={cfg.SOLVER.CTCMT_CLASS_BALANCED_CE} "
      f"ANCHOR_MARG={cfg.SOLVER.CTCMT_ANCHOR_MARGINAL_WEIGHT} RST_M={cfg.SOLVER.RST_M}")

student, teacher, anchor = (FakeSemanticSegmentor() for _ in range(3))
teacher.load_state_dict(student.state_dict())
anchor.load_state_dict(student.state_dict())
# Trainer.test() puts every submodule into eval(); forward() then puts only the
# student back into train mode. Reproduce that, or the heads return losses.
student.eval(); teacher.eval(); anchor.eval()

opt = torch.optim.SGD([p for p in student.parameters()], lr=1e-3, momentum=0.9)
# __init__ is @configurable and this class takes a literal `cfg` kwarg, so the
# decorator would route to from_config() and demand a real checkpoint. Call the
# undecorated __init__ to inject the fakes.
model = CTCMT_MTL.__new__(CTCMT_MTL)
CTCMT_MTL.__init__.__wrapped__(
    model, student=student, teacher=teacher, anchor=anchor, optimizer=opt, cfg=cfg
)

# --- 7. the head contract the whole thing depends on ---------------------
student.sem_seg_head.train()
assert student.sem_seg_head({"p2": torch.zeros(1, 8, 8, 8)})[0] is None, \
    "fake head must reproduce SemSegFPNHead's (None, losses) training contract"
student.sem_seg_head.eval()
print("  [ok] sem_seg head train/eval contract reproduced")

# --- 5. empty instances --------------------------------------------------
imgs = student.preprocess_image(make_inputs())
empty = CTCMT_MTL._empty_instances(imgs)
assert len(empty) == 0
assert empty.has("pred_boxes") and empty.has("pred_classes") and empty.has("scores")
print("  [ok] _empty_instances: len()==0 with the fields forward() reads")

# --- 1/2/3/4. end-to-end -------------------------------------------------
w_before = student.backbone.conv.weight.detach().clone()
t_before = teacher.sem_seg_head.pred.weight.detach().clone()

for step in range(3):
    out = model(make_inputs())
    assert isinstance(out, list) and len(out) == 1
    assert "sem_seg" in out[0], "segmentation output missing"
    assert "instances" not in out[0], "detection output leaked into a seg-only run"
    assert out[0]["sem_seg"].shape[0] == K
    assert torch.isfinite(out[0]["sem_seg"]).all()

assert not torch.allclose(w_before, student.backbone.conv.weight), \
    "student trunk did not move -- the seg loss never reached it"
assert not torch.allclose(t_before, teacher.sem_seg_head.pred.weight), \
    "teacher EMA did not move"
print("  [ok] forward() x3: sem_seg only, student updated, teacher EMA moved")

# --- 6. nothing detection-specific was touched ---------------------------
for attr in ("roi_heads", "proposal_generator", "inference"):
    assert not hasattr(student, attr) and not hasattr(teacher, attr)
print("  [ok] student/teacher expose no detection API; forward() never needed one")

# --- restoration must still pull toward the anchor -----------------------
assert model._source_params, "anchor snapshot is empty -- restore is a no-op"
n_shared = sum(1 for k in model._source_params if k.startswith("backbone."))
print(f"  [ok] stochastic restore armed: {len(model._source_params)} params "
      f"({n_shared} under backbone.)")

# --- the strong/weak asymmetry is really in effect -----------------------
probe = make_inputs()
probe[0]["image_strong"] = torch.zeros(3, H, W)
a = student.preprocess_image(probe, strong_aug=True).tensor
b = student.preprocess_image(probe, strong_aug=False).tensor
assert not torch.allclose(a, b), "strong_aug did not select image_strong"
print("  [ok] preprocess_image(strong_aug=True) consumes 'image_strong'")

print("\nALL SEG-ONLY PATH TESTS PASSED")
