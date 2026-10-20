#!/usr/bin/env python3
"""Unit test for the E9 additions.

  1. _fisher_restore_mask selects ~perc of entries and prefers low-gradient ones.
  2. Degenerate perc values behave.
  3. preprocess_image(strong_aug=True) reads "image_strong" and differs from the
     weak view, while keeping identical shape (geometry-preserving).
"""
import torch

from detectron2.modeling.meta_arch.ctcmt_mtl import _fisher_restore_mask

torch.manual_seed(0)

# --- 1. selection rate and low-importance preference -----------------------
fisher = torch.rand(200, 200)
for perc in (0.001, 0.01, 0.1):
    rates, biases = [], []
    for _ in range(20):
        mask = _fisher_restore_mask(fisher, perc)
        rates.append(mask.mean().item())
        sel = fisher[mask.bool()]
        if sel.numel():
            biases.append(sel.mean().item())
    rate = sum(rates) / len(rates)
    assert abs(rate - perc) < max(0.25 * perc, 1e-4), (perc, rate)
    mean_sel = sum(biases) / len(biases)
    assert mean_sel < fisher.mean().item(), (perc, mean_sel, fisher.mean().item())
    print(f"  perc={perc:<6} rate={rate:.5f}  mean|g^2| selected={mean_sel:.4f} "
          f"(overall {fisher.mean().item():.4f})")

# --- 2. degenerate cases ----------------------------------------------------
assert _fisher_restore_mask(fisher, 0.0).sum() == 0
assert _fisher_restore_mask(fisher, 1.0).mean() == 1.0
zero = torch.zeros(10, 10)
m = _fisher_restore_mask(zero, 0.5)
assert torch.isfinite(m).all()

# --- 3. strong-aug view -----------------------------------------------------
from detectron2.modeling.meta_arch import GeneralizedRCNN  # noqa: E402


class _Stub(GeneralizedRCNN):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.register_buffer("pixel_mean", torch.zeros(3, 1, 1), False)
        self.register_buffer("pixel_std", torch.ones(3, 1, 1), False)

    @property
    def backbone(self):
        class _B:
            size_divisibility = 32
            padding_constraints = {}
        return _B()


stub = _Stub()
weak = torch.randint(0, 255, (3, 64, 64)).float()
strong = torch.randint(0, 255, (3, 64, 64)).float()
batch = [{"image": weak, "image_strong": strong}]

w = stub.preprocess_image(batch, strong_aug=False)
s = stub.preprocess_image(batch, strong_aug=True)
assert w.tensor.shape == s.tensor.shape, (w.tensor.shape, s.tensor.shape)
assert not torch.allclose(w.tensor, s.tensor), "strong view is identical to weak"
assert torch.allclose(s.tensor[0, :, :64, :64], strong)

print("E9 UNIT TEST PASSED")
print("Verified: Fisher mask rate + low-importance bias; strong_aug reads image_strong.")
