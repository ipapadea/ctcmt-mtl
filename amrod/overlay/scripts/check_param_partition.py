#!/usr/bin/env python3
"""Structural check for the gradient-surgery design.

Two ways `_backward_and_combine` could be silently wrong:
  1. a trainable param outside the optimizer -> `zero_grad` between passes would
     not clear it, so pass-2 grads would accumulate on top of pass-1;
  2. a trainable param matching none of shared/det/seg -> it would be treated as
     head-specific by default and silently escape the surgery.

  python scripts/check_param_partition.py [CONFIG]
"""
import sys

from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.ctcmt_mtl import _SHARED_PARAM_PREFIXES
from detectron2.solver import build_optimizer

cfg = get_cfg()
cfg.merge_from_file(sys.argv[1] if len(sys.argv) > 1
                    else "detectron2/configs/Cityscapes/ctcmt_e16_protectedgrad.yaml")
cfg.MODEL.DEVICE = "cpu"

student = META_ARCH_REGISTRY.get(cfg.MODEL.CTCMT_STUDENT_META_ARCH)(cfg)
student.roi_heads.mask_on = False
opt = build_optimizer(cfg, student)

in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
train = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
missing = [n for n, p in train if id(p) not in in_opt]

shared = [n for n, _ in train if any(n.startswith(p) for p in _SHARED_PARAM_PREFIXES)]
det = [n for n, _ in train if n.startswith(("proposal_generator.", "roi_heads."))]
seg = [n for n, _ in train if n.startswith("sem_seg_head.")]
other = [n for n, _ in train if n not in set(shared) | set(det) | set(seg)]

print(f"trainable params : {len(train)}")
print(f"in optimizer     : {sum(1 for _, p in train if id(p) in in_opt)}")
print(f"NOT in optimizer : {len(missing)}  {missing[:10]}")
print(f"partition        : shared {len(shared)} | det {len(det)} | seg {len(seg)} "
      f"| unclassified {len(other)}")
print(f"unclassified     : {other}")

bad = bool(missing) or bool(other)
print("FAIL" if bad else "OK - every trainable param is optimizer-managed and classified")
sys.exit(1 if bad else 0)
