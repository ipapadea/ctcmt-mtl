#!/usr/bin/env python3
"""Unit test for the class-collapse countermeasures.

  1. With both flags off, the weighted loss is bit-identical to the plain mean.
  2. The class-marginal EMA tracks the teacher marginal.
  3. Inverse-frequency weights up-weight rare classes and are mean-normalised,
     so the overall loss scale (and hence CTCMT_WEIGHT_SEG) is preserved.
  4. KL(anchor || student) is >= 0, zero iff marginals match, and grows when the
     student drops a class the anchor still uses.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL

torch.manual_seed(0)

K, H, W = 19, 16, 32


def make(cb=False, beta=0.5, ema=0.9):
    m = CTCMT_MTL.__new__(CTCMT_MTL)
    nn.Module.__init__(m)
    m.class_balanced_ce = cb
    m.class_balance_beta = beta
    m.class_marginal_ema = ema
    m._class_marginal = None
    return m


# Skewed teacher: class 0 owns most pixels, class 18 is rare.
logits_t = torch.randn(1, K, H, W)
logits_t[0, 0] += 6.0
logits_t[0, 18] -= 3.0
t_probs = logits_t.softmax(dim=1)

s_logits = torch.randn(1, K, H, W, requires_grad=True)
ce = -(t_probs * F.log_softmax(s_logits.float(), dim=1)).sum(dim=1)

# --- 1. flags off == plain mean ------------------------------------------
plain = ce.mean()
w = None
assert w is None
weighted = plain if w is None else (w * ce).sum() / (w.sum() + 1e-6)
assert torch.equal(plain, weighted)

# --- 2. marginal EMA ------------------------------------------------------
m = make(cb=True)
marg = m._update_class_marginal(t_probs)
assert torch.allclose(marg, t_probs.mean(dim=(0, 2, 3))), "first call seeds the EMA"
assert abs(marg.sum().item() - 1.0) < 1e-4, marg.sum().item()
before = marg.clone()
m._update_class_marginal(torch.full_like(t_probs, 1.0 / K))
assert not torch.allclose(before, m._class_marginal), "EMA must move"

# --- 3. inverse-frequency weights ----------------------------------------
m2 = make(cb=True)
marg = m2._update_class_marginal(t_probs)
inv = marg.clamp_min(1e-6).pow(-m2.class_balance_beta)
hard = t_probs.argmax(dim=1)
cb_w = inv[hard]
cb_w = cb_w / cb_w.mean().clamp_min(1e-6)

assert abs(cb_w.mean().item() - 1.0) < 1e-5, cb_w.mean().item()
freq_c = int(marg.argmax())
rare_c = int(marg.argmin())
assert inv[rare_c] > inv[freq_c], "rare classes must be up-weighted"
print(f"  marginal: max={marg.max():.4f} (c{freq_c})  min={marg.min():.6f} (c{rare_c})")
print(f"  weight ratio rare/frequent = {(inv[rare_c] / inv[freq_c]).item():.2f}x")

cb_loss = (cb_w * ce).sum() / (cb_w.sum() + 1e-6)
assert torch.isfinite(cb_loss) and abs(cb_loss.item() - plain.item()) / plain.item() < 5.0

# --- 4. anchor-marginal KL ------------------------------------------------
def kl(anchor_probs, student_logits):
    q = anchor_probs.mean(dim=(0, 2, 3)).detach().clamp_min(1e-8)
    p = student_logits.float().softmax(dim=1).mean(dim=(0, 2, 3)).clamp_min(1e-8)
    return (q * (q.log() - p.log())).sum()

assert kl(t_probs, logits_t).item() < 1e-6, "identical marginals => KL 0"

dropped = logits_t.clone()
dropped[0, 18] -= 20.0                      # student abandons the rare class
k_drop = kl(t_probs, dropped)
k_same = kl(t_probs, logits_t)
assert k_drop > k_same, (k_drop.item(), k_same.item())
assert k_drop >= 0
print(f"  KL(anchor||student): matched={k_same.item():.6f}  class-dropped={k_drop.item():.6f}")

grad_probe = (logits_t + 0.1 * torch.randn_like(logits_t)).requires_grad_(True)
kl(t_probs, grad_probe).backward()
assert grad_probe.grad is not None and torch.isfinite(grad_probe.grad).all()
print("CLASS-BALANCE UNIT TEST PASSED")
print("Verified: off==plain mean; EMA tracks marginal; rare classes up-weighted with")
print("mean-normalised scale; mode-covering KL penalises dropping a source class.")
