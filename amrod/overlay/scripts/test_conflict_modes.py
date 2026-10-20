#!/usr/bin/env python3
"""Unit test for the S1/S2/S3/S5 shared-trunk gradient combination.

Builds a tiny two-head model whose "shared trunk" is named ``backbone.*`` so
the real prefix logic is exercised, then checks every conflict mode against an
explicit reference computed on flattened gradient vectors.

Checks:
  1. mode "none" reproduces the plain joint backward exactly.
  2. protect_det: no-op when the tasks agree; exact PCGrad-style projection
     when they conflict; g_det is never modified; heads keep their own grads.
  3. hard_decouple: shared trunk gets g_det only on conflict, full sum otherwise.
  4. dyn_weight: gradient equals g_det + max(0, cos) * g_aux everywhere.
  5. cagrad: matches the closed form and is symmetric in the two tasks.
  6. Component diagnostics report the right per-loss cosines and are logging
     only -- they do not change the applied gradient.

Run inside the image:
  python scripts/test_conflict_modes.py
"""
import math

import torch
import torch.nn as nn

from detectron2.modeling.meta_arch.ctcmt_mtl import CTCMT_MTL, _cagrad_weight

torch.manual_seed(0)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4, bias=False)
        self.roi_heads = nn.Linear(4, 2, bias=False)
        self.sem_seg_head = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        h = self.backbone(x)
        return self.roi_heads(h), self.sem_seg_head(h)


class FakeOpt:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None


REF_STATE = {k: v.clone() for k, v in Tiny().state_dict().items()}


def make(mode, grad_diag=False, alpha=0.5):
    m = CTCMT_MTL.__new__(CTCMT_MTL)
    nn.Module.__init__(m)
    m.student = Tiny()
    m.student.load_state_dict(REF_STATE)  # every arm must see identical weights
    m.optimizer = FakeOpt(m.student.parameters())
    m.conflict_mode = mode
    m.cagrad_alpha = alpha
    m.grad_diag = grad_diag
    m.grad_diag_every = 1
    m.route_scope = "shared"
    m.iter = 0
    m.aux_trunk_lambda = 0.0
    m._route_lambda = 0.0
    m.adaptive_routing = False
    m.adaptive_routing_beta = 0.98
    m.adaptive_routing_ema = 0.99
    m._seg_agree_ema = None
    m._seg_agree_max = 0.0
    m._param_index = None
    m._routed_set = None
    m._shared_names = None
    m._conflict_stats = {
        "steps": 0, "both": 0, "conflicts": 0, "projected": 0,
        "cos_sum": 0.0, "gamma_sum": 0.0, "w_det_sum": 0.0,
    }
    return m


def losses(model, x, flip):
    """flip<0 makes the seg loss pull the trunk against the det loss."""
    det, seg = model(x)
    return {
        "det/cls": det.pow(2).mean(),
        "seg/soft_ce": flip * seg.mean(),
        "ctcr": 0.1 * seg.pow(2).mean(),
    }


def grads_of(m, x, flip, group):
    """Reference per-group gradients as a flat dict, computed independently."""
    m.optimizer.zero_grad()
    ld = losses(m.student, x, flip)
    if group == "det":
        total = ld["det/cls"]
    elif group == "aux":
        total = ld["seg/soft_ce"] + ld["ctcr"]
    else:
        total = ld[group]
    total.backward()
    out = {n: (p.grad.clone() if p.grad is not None else None)
           for n, p in m.student.named_parameters()}
    m.optimizer.zero_grad()
    return out


def flat(d, names):
    return torch.cat([d[n].reshape(-1) for n in names if d.get(n) is not None])


X = torch.randn(6, 4)
SHARED = ["backbone.weight"]
HEADS = ["roi_heads.weight", "sem_seg_head.weight"]


def run(mode, flip, **kw):
    m = make(mode, **kw)
    gd = grads_of(m, X, flip, "det")
    ga = grads_of(m, X, flip, "aux")
    m.optimizer.zero_grad()
    diag = m._backward_and_combine(losses(m.student, X, flip))
    got = {n: (p.grad.clone() if p.grad is not None else None)
           for n, p in m.student.named_parameters()}
    return m, gd, ga, got, diag


for FLIP in (+1.0, -1.0):
    d_ref = grads_of(make("none"), X, FLIP, "det")
    a_ref = grads_of(make("none"), X, FLIP, "aux")
    gdv, gav = flat(d_ref, SHARED), flat(a_ref, SHARED)
    dot = float(gdv @ gav)
    nd, na = float(gdv.norm()), float(gav.norm())
    cos = dot / (nd * na + 1e-12)
    conflict = dot < 0.0
    print(f"\n=== flip={FLIP:+.0f}  dot={dot:+.6f}  cos={cos:+.4f}  conflict={conflict}")

    # --- 1. none == plain joint backward ----------------------------------
    _, _, _, got, _ = run("none", FLIP)
    m0 = make("none")
    m0.optimizer.zero_grad()
    sum(losses(m0.student, X, FLIP).values()).backward()
    ref_joint = {n: p.grad.clone() for n, p in m0.student.named_parameters()}
    for n in SHARED + HEADS:
        assert torch.allclose(got[n], ref_joint[n], atol=1e-7), (n, got[n], ref_joint[n])
    print("  [ok] none == joint backward")

    # --- 2. protect_det ----------------------------------------------------
    m, gd, ga, got, diag = run("protect_det", FLIP)
    aux_proj = gav - (dot / (nd ** 2 + 1e-12)) * gdv if conflict else gav
    assert torch.allclose(flat(got, SHARED), gdv + aux_proj, atol=1e-6)
    assert abs(float((gdv @ aux_proj)) if conflict else 0.0) < 1e-5, "not orthogonal"
    assert diag["projected"] is conflict
    for n in HEADS:  # heads must keep their untouched task gradients
        ref = (gd[n] if gd[n] is not None else 0) + (ga[n] if ga[n] is not None else 0)
        assert torch.allclose(got[n], ref, atol=1e-7), n
    assert abs(diag["cos"] - cos) < 1e-5
    print(f"  [ok] protect_det  projected={diag['projected']} "
          f"|g_aux|={diag['g_aux']:.5f} -> |g_aux_proj|={diag['g_aux_proj']:.5f}")

    # --- 3. hard_decouple --------------------------------------------------
    m, gd, ga, got, diag = run("hard_decouple", FLIP)
    want = gdv if conflict else gdv + gav
    assert torch.allclose(flat(got, SHARED), want, atol=1e-6)
    assert diag["decoupled"] is conflict
    for n in HEADS:
        ref = (gd[n] if gd[n] is not None else 0) + (ga[n] if ga[n] is not None else 0)
        assert torch.allclose(got[n], ref, atol=1e-7), n
    print(f"  [ok] hard_decouple  decoupled={diag['decoupled']}")

    # --- 4. dyn_weight -----------------------------------------------------
    m, gd, ga, got, diag = run("dyn_weight", FLIP)
    gamma = max(0.0, cos)
    assert abs(diag["gamma"] - gamma) < 1e-6
    assert torch.allclose(flat(got, SHARED), gdv + gamma * gav, atol=1e-6)
    for n in HEADS:  # loss reweighting => heads are scaled too
        ref = (gd[n] if gd[n] is not None else 0) + gamma * (ga[n] if ga[n] is not None else 0)
        assert torch.allclose(got[n], ref, atol=1e-6), n
    print(f"  [ok] dyn_weight  gamma={gamma:.4f}")

    # --- 4b. aux_head_only: trunk sees ONLY g_det, on every step ----------
    m, gd, ga, got, diag = run("aux_head_only", FLIP)
    assert torch.allclose(flat(got, SHARED), gdv, atol=1e-6), "trunk must equal g_det"
    assert diag["decoupled"] is True, "routing must be unconditional, not conflict-gated"
    for n in HEADS:  # heads keep their full, unmodified task gradients
        ref = (gd[n] if gd[n] is not None else 0) + (ga[n] if ga[n] is not None else 0)
        assert torch.allclose(got[n], ref, atol=1e-7), n
    print(f"  [ok] aux_head_only  trunk==g_det, heads untouched (conflict={conflict})")

    # --- 4c. aux_head_only with NO detection loss: trunk must not move -----
    # ~25% of real steps have no surviving pseudo-box. If the routing lapsed
    # there, the seg gradient would hit the trunk at full strength.
    m = make("aux_head_only")
    ld = losses(m.student, X, FLIP)
    ld.pop("det/cls")
    m.optimizer.zero_grad()
    m._backward_and_combine(ld)
    assert m.student.backbone.weight.grad is None, "seg leaked into the trunk"
    assert m.student.sem_seg_head.weight.grad is not None, "seg head must still adapt"
    print("  [ok] aux_head_only  det-absent step: trunk untouched, seg head adapts")

    # --- 4d. aux_head_only with a routing fraction lambda ------------------
    # lambda interpolates: 0 == S6, 1 == the plain joint update. The heads must
    # keep their FULL gradient at every lambda.
    for lam in (0.0, 0.25, 1.0):
        m = make("aux_head_only")
        m.aux_trunk_lambda = m._route_lambda = lam
        gd = grads_of(m, X, FLIP, "det")
        ga = grads_of(m, X, FLIP, "aux")
        m.optimizer.zero_grad()
        m._backward_and_combine(losses(m.student, X, FLIP))
        got = {n: (p.grad.clone() if p.grad is not None else None)
               for n, p in m.student.named_parameters()}
        want = gdv + lam * gav
        assert torch.allclose(flat(got, SHARED), want, atol=1e-6), lam
        for n in HEADS:
            ref = (gd[n] if gd[n] is not None else 0) + (ga[n] if ga[n] is not None else 0)
            assert torch.allclose(got[n], ref, atol=1e-7), (lam, n)
    print("  [ok] aux_head_only lambda in {0, 0.25, 1}: trunk == g_det + lam*g_aux, "
          "heads full")

    # lambda=1 must equal the plain joint backward, and lambda=0 must equal S6.
    m1 = make("aux_head_only"); m1.aux_trunk_lambda = m1._route_lambda = 1.0
    m1.optimizer.zero_grad(); m1._backward_and_combine(losses(m1.student, X, FLIP))
    m0 = make("none")
    m0.optimizer.zero_grad(); sum(losses(m0.student, X, FLIP).values()).backward()
    for n in SHARED + HEADS:
        a = dict(m1.student.named_parameters())[n].grad
        b = dict(m0.student.named_parameters())[n].grad
        assert torch.allclose(a, b, atol=1e-6), n
    print("  [ok] lambda=1 reproduces the plain joint update exactly")

    # --- 4e. det-absent step scales the trunk by lambda, not by 1 ----------
    for lam, expect_none in ((0.0, True), (0.25, False)):
        m = make("aux_head_only")
        m.aux_trunk_lambda = m._route_lambda = lam
        ld = losses(m.student, X, FLIP); ld.pop("det/cls")
        ref = grads_of(m, X, FLIP, "aux")
        m.optimizer.zero_grad()
        m._backward_and_combine(ld)
        g = m.student.backbone.weight.grad
        if expect_none:
            assert g is None, "lambda=0 must leave the trunk untouched"
        else:
            assert torch.allclose(g, ref["backbone.weight"] * lam, atol=1e-6), lam
        assert m.student.sem_seg_head.weight.grad is not None
    print("  [ok] det-absent step: trunk scaled by lambda, seg head still adapts")

    # --- 5. cagrad ---------------------------------------------------------
    ALPHA = 0.5
    m, gd, ga, got, diag = run("cagrad", FLIP, alpha=ALPHA)
    x = _cagrad_weight(nd ** 2, dot, na ** 2, ALPHA)
    assert abs(x - diag["w_det"]) < 1e-6
    gw = x * gdv + (1.0 - x) * gav
    g0 = 0.5 * (gdv + gav)
    coef = ALPHA * float(g0.norm()) / (float(gw.norm()) + 1e-8)
    want = 2.0 * (g0 + coef * gw) / (1.0 + ALPHA ** 2)
    assert torch.allclose(flat(got, SHARED), want, atol=1e-5), (flat(got, SHARED), want)
    # x must minimise the CAGrad objective over [0, 1].
    def F(t):
        gt = t * gdv + (1.0 - t) * gav
        return float(gt @ g0) + ALPHA * float(g0.norm()) * float(gt.norm())
    best = min((i / 500.0 for i in range(501)), key=F)
    assert F(x) <= F(best) + 1e-6, (x, best, F(x), F(best))
    for n in HEADS:
        ref = (gd[n] if gd[n] is not None else 0) + (ga[n] if ga[n] is not None else 0)
        assert torch.allclose(got[n], ref, atol=1e-7), n
    print(f"  [ok] cagrad  w_det={x:.4f} w_aux={1 - x:.4f} "
          f"c_det={diag['c_det']:.4f} c_aux={diag['c_aux']:.4f}")

    # --- 6. component diagnostics are logging only -------------------------
    m_a, _, _, got_a, _ = run("protect_det", FLIP, grad_diag=False)
    m_b, _, _, got_b, diag_b = run("protect_det", FLIP, grad_diag=True)
    for n in SHARED + HEADS:
        assert torch.allclose(got_a[n], got_b[n], atol=1e-6), n
    for comp in ("seg", "ctcr"):
        g_c = flat(grads_of(make("none"), X, FLIP, {"seg": "seg/soft_ce", "ctcr": "ctcr"}[comp]), SHARED)
        want_cos = float(gdv @ g_c) / (nd * float(g_c.norm()) + 1e-12)
        assert abs(diag_b[f"cos_det_{comp}"] - want_cos) < 1e-5, comp
    assert set(diag_b["blocks"]) == {"fpn"}  # "backbone.weight" is not a resN block
    print(f"  [ok] graddiag  cos_det_seg={diag_b['cos_det_seg']:+.4f} "
          f"cos_det_ctcr={diag_b['cos_det_ctcr']:+.4f}  (update unchanged)")

    # --- 7. REGRESSION: a detached constant component must not raise -------
    # _supcon_loss returns features.new_zeros(()) when a pseudo-box set has no
    # positive pairs. Summed with live losses that is harmless, but backwarding
    # it alone raises "element 0 of tensors does not require grad" -- this
    # killed four 6-hour runs on 2026-09-12.
    for mode in ("protect_det", "cagrad", "hard_decouple", "dyn_weight", "aux_head_only"):
        m = make(mode, grad_diag=True)
        ld = losses(m.student, X, FLIP)
        ld["ctcl"] = torch.zeros(())            # detached constant, no grad_fn
        m.optimizer.zero_grad()
        m._backward_and_combine(ld)             # must not raise
        got_const = {n: (p.grad.clone() if p.grad is not None else None)
                     for n, p in m.student.named_parameters()}
        # A zero-gradient component must not change the applied update.
        m2 = make(mode, grad_diag=True)
        m2.optimizer.zero_grad()
        m2._backward_and_combine(losses(m2.student, X, FLIP))
        for n in SHARED + HEADS:
            ref = {nm: (p.grad if p.grad is not None else None)
                   for nm, p in m2.student.named_parameters()}[n]
            assert torch.allclose(got_const[n], ref, atol=1e-6), (mode, n)
    print("  [ok] constant (no grad_fn) aux component: no raise, update unchanged")

    # --- 8. every loss constant -> no-op, still no raise -------------------
    m = make("protect_det", grad_diag=True)
    m.optimizer.zero_grad()
    m._backward_and_combine({"det/cls": torch.zeros(()), "ctcl": torch.zeros(())})
    print("  [ok] all-constant loss dict handled")

print("\nALL CONFLICT-MODE TESTS PASSED")

# --- 9. adaptive routing gate (Option 3) ---------------------------------
# The trunk should stay open while the seg teacher holds its agreement with the
# frozen anchor, and close once that agreement decays past beta of its own peak.
g = make("aux_head_only")
g.aux_trunk_lambda = 0.25
g.adaptive_routing = True
g.adaptive_routing_beta = 0.98
g.adaptive_routing_ema = 0.9          # fast EMA so the test is short

lam_first = g._update_route_lambda(0.90)
assert lam_first == 0.25, "first step must open the trunk"
for _ in range(20):                    # hold steady -> stays open
    lam = g._update_route_lambda(0.90)
assert lam == 0.25, f"steady agreement must keep lambda open, got {lam}"
peak = g._seg_agree_max
for _ in range(50):                    # teacher drifts away from source
    lam = g._update_route_lambda(0.40)
assert lam == 0.0, f"drifting agreement must close the trunk, got {lam}"
assert g._seg_agree_max == peak, "running max must not follow the decay"
for _ in range(200):                   # recovery re-opens it
    lam = g._update_route_lambda(0.95)
assert lam == 0.25, "recovered agreement must re-open the trunk"
print(f"  [ok] adaptive routing: open at steady agreement, closes on drift "
      f"(peak {peak:.3f}), re-opens on recovery")

# lambda=0 arms must ignore the gate entirely.
g0 = make("aux_head_only")
g0.aux_trunk_lambda = 0.0
g0.adaptive_routing = True
assert g0._update_route_lambda(0.99) == 0.0, "lambda=0 must stay 0 whatever the gate"
print("  [ok] adaptive routing cannot raise lambda above CTCMT_AUX_TRUNK_LAMBDA")

print("\nALL ROUTING TESTS PASSED")
