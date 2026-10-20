"""Source-side checks for DAT-BF (same-source Panoptic-FPN port).

Uses only Cityscapes *source-domain* train images and one synthetic noise image;
no target-stream data and no ground truth. Run inside the amrod container, e.g.

  python tools/dat_bf_source_checks.py --config configs/Cityscapes/newsrc_dat_bf_csc.yaml \
      --out /scratch/dat_bf_source_checks --images IMG1.png IMG2.png ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

# Test harness only: the audit on/off comparison needs bitwise-reproducible kernels
# (run with CUBLAS_WORKSPACE_CONFIG=:4096:8). Nothing in the model sets these flags.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch.dat_pfn import DATDivergenceError

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def load_inputs(paths, cfg, size_wh=(1000, 500)):
    rng = np.random.RandomState(0)
    arrays = []
    for p in paths:
        arrays.append(np.asarray(Image.open(p).convert("RGB").resize(size_wh, Image.BILINEAR)))
    arrays.append(rng.randint(0, 256, size=(size_wh[1], size_wh[0], 3)).astype(np.uint8))  # synthetic noise
    out = []
    for i, a in enumerate(arrays):
        if cfg.INPUT.FORMAT == "BGR":
            a = a[:, :, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()
        out.append({"image": t, "height": size_wh[1], "width": size_wh[0], "file_name": f"source_check_{i}"})
    return out


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.OUTPUT_DIR = args.out
    cfg.freeze()
    model = build_model(cfg)
    inputs = load_inputs(args.images, cfg)
    dev = model.device
    print("device:", dev)

    # ---- static identity / eligibility ------------------------------------ #
    check("canonical MD5 asserted at build", cfg.SOLVER.DAT_EXPECTED_SOURCE_MD5 == "5f63c7578cb925552fd1b9d69980937a")
    check("eligible = 92 tensors / 28,425,555 elements",
          (model.eligible_tensors, model.eligible_elements) == (92, 28425555))
    check("cap = 2,842,555 elements, k = 1", (model.cap_elements, model.k) == (2842555, 1))
    check("all student tensors trainable (FREEZE_AT overridden)", all(p.requires_grad for p in model.params))
    check("teacher and anchor frozen",
          not any(p.requires_grad for p in model.teacher.parameters())
          and not any(p.requires_grad for p in model.anchor.parameters()))
    check("no detection modules / checkpoint det keys not instantiated",
          not any(n.startswith(("roi_heads", "proposal_generator")) for n, _ in model.student.named_modules()))
    same = all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in
               zip(model.student.parameters(), model.teacher.parameters(), model.anchor.parameters()))
    check("student == teacher == anchor bitwise at init", same)
    src = [p.detach().clone() for p in model.anchor.parameters()]
    tea0 = [p.detach().clone() for p in model.teacher.parameters()]

    # ---- views: index mapping and anchor shortcut equivalence --------------- #
    order = [(s, f) for s, f, _, _ in model._aug_ops]
    check("14 views, scale-major/flip-minor; idx 4,6,8 = (1.0,F),(1.25,F),(1.5,F)",
          len(order) == 14 and [order[i] for i in (4, 6, 8)] == [(1.0, False), (1.25, False), (1.5, False)], str(order[4:10]))
    images = model._preprocess(inputs[:1])
    x = images.tensor
    check("padding present in check input", tuple(images.image_sizes[0]) != tuple(x.shape[-2:]),
          f"valid={tuple(images.image_sizes[0])} padded={tuple(x.shape[-2:])}")
    with torch.no_grad():
        full = [p.max(dim=1).values for _, _, p in model._view_probs(model.anchor, x, model._aug_ops)]
        ref_score = (full[4] + full[6] + full[8]) / 3
        maps = model._anchor_maps(x)
        short_score = torch.stack(maps).mean(0)
        check("3-view anchor shortcut == 14-view indices 4,6,8 (exact)",
              float((ref_score - short_score).abs().max()) == 0.0 and torch.equal(full[4], maps[0]))
        # plain-view teacher logits equal the stock SemanticSegmentor head output
        feats = model.teacher.backbone(x)
        stock, _ = model.teacher.sem_seg_head(feats, None)
        mine = torch.nn.functional.interpolate(model._logits(model.teacher, x), size=x.shape[-2:],
                                               mode="bilinear", align_corners=False)
        check("plain-view logits == stock SemSegFPNHead eval output",
              float((stock - mine).abs().max()) < 1e-4, f"max|diff|={float((stock - mine).abs().max()):.2e}")

    # ---- frames --------------------------------------------------------------- #
    pre_plain = []
    logs = []
    ema_checked = False
    for k, inp in enumerate(inputs):
        with torch.no_grad():
            im = model._preprocess([inp]).tensor
            pre_plain.append(model._teacher_probs(im)[1].argmax(1).cpu())
            tea_before = [p.detach().clone() for p in model.teacher.parameters()]
        out = model([inp])
        logs.append(json.loads(open(model._frame_log_path).read().strip().splitlines()[-1]))
        check(f"frame {k+1}: output sem_seg shape (C,H,W) float", out[0]["sem_seg"].shape == (19, 500, 1000))
        check(f"frame {k+1}: all .grad None after frame", all(p.grad is None for p in model.params))
        if not ema_checked and logs[-1]["step_taken"]:
            idx = model.selector.selected_sorted[0]
            expect = 0.999 * tea_before[idx] + 0.001 * model.params[idx].detach()
            err = float((list(model.teacher.parameters())[idx] - expect).abs().max())
            check("EMA teacher = 0.999*teacher + 0.001*student after the step", err < 1e-6, f"max err {err:.2e}")
            ema_checked = True

    sel = model.selector.selected_sorted
    sel_set = set(sel)
    check("selection non-empty", len(sel) > 0, f"{len(sel)} tensors, {model.selector.selected_elements} elements "
          f"({100*model.selector.selected_elements/model.eligible_elements:.3f}%)")
    check("cap never exceeded", model.selector.selected_elements <= model.cap_elements)
    check("some frame took an optimizer step", any(r["step_taken"] for r in logs))
    stu = list(model.student.parameters())
    changed = [i for i in range(len(stu)) if not torch.equal(stu[i].detach(), src[i])]
    check("student tensors that changed are a subset of the selected set", set(changed) <= sel_set,
          f"changed={len(changed)} selected={len(sel)}")
    check("selected tensors changed", any(r["step_taken"] for r in logs) and all(i in set(changed) for i in sel))
    check("non-selected student tensors bit-identical to source", all(torch.equal(stu[i].detach(), src[i])
                                                                      for i in range(len(stu)) if i not in sel_set))
    tea = list(model.teacher.parameters())
    nonsel_dev = max([float((tea[i].detach() - src[i]).abs().max()) for i in range(len(tea)) if i not in sel_set] or [0.0])
    check("non-selected teacher tensors within float rounding of source", nonsel_dev < 1e-6, f"max|diff|={nonsel_dev:.2e}")
    finite = all(torch.isfinite(p).all() for p in stu) and all(torch.isfinite(p).all() for p in tea)
    check("no NaN/Inf in student/teacher", bool(finite))
    delta_max = max(r.get("delta_max_abs", 0.0) for r in logs)
    check("first-step update magnitude <= lr (fresh Adam)", delta_max <= cfg.SOLVER.DAT_LR * (1 + 1e-3),
          f"max|delta|={delta_max:.3e} lr={cfg.SOLVER.DAT_LR}")

    # pre-update claims: primary pseudo-label / secondary plain view saved before the update
    pd = os.path.join(args.out, "dat_bf_predictions", "plain_view")
    eq = all(np.array_equal(np.array(Image.open(os.path.join(pd, f"{i+1:06d}.png"))),
                            pre_plain[i][0, :500, :1000].numpy().astype(np.uint8)) for i in range(len(inputs)))
    check("secondary plain-view prediction == pre-update teacher scale-1.0 prediction", eq)
    check("primary prediction files written", all(os.path.exists(os.path.join(args.out, "dat_bf_predictions", "primary",
                                                                              f"{i+1:06d}.png")) for i in range(len(inputs))))

    # empty-selected-set / group paths seen in the log
    print("outcomes seen:", sorted({r["dsp"]["outcome"] for r in logs} | {r["trp"]["outcome"] for r in logs}))
    print("group fractions per frame (dsp, trp, mid):",
          [(round(r["frac_dsp"], 3), round(r["frac_trp"], 3), round(r["frac_mid"], 3)) for r in logs])

    # ---- audit invariants (target-path instrumentation) and audit on/off equivalence ---- #
    check("audit: changed_nonselected_student_tensors == 0 on every frame",
          all(r["audit_changed_nonselected_student_tensors"] == 0 and r["audit_changed_nonselected_student_max_abs"] == 0.0
              for r in logs))
    check("audit: non-selected student tensors identical to source on every frame",
          all(r["audit_nonselected_student_vs_source_changed_tensors"] == 0 for r in logs))
    check("audit: EMA consistency within tolerance on every frame", all(r["audit_ema_consistent"] for r in logs),
          f"max ratio={max(r['audit_ema_residual_ratio_max'] for r in logs):.3f}, "
          f"max resid={max(r['audit_ema_residual_max_abs'] for r in logs):.2e}")
    check("audit: unchanged selected tensors logged",
          all(0 <= r["audit_unchanged_selected_student_tensors"] <= r["selected_tensors"] for r in logs))
    print("audit: non-selected teacher dev max abs / ratio(2^-23|src|):",
          max(r["audit_nonselected_teacher_dev_max_abs"] for r in logs),
          max(r["audit_nonselected_teacher_dev_ratio_max"] for r in logs))
    state_a = [p.detach().clone() for p in model.student.parameters()] + [p.detach().clone() for p in model.teacher.parameters()]
    cfg_b = cfg.clone()
    cfg_b.defrost()
    cfg_b.SOLVER.DAT_AUDIT = False
    cfg_b.OUTPUT_DIR = args.out.rstrip("/") + "_noaudit"
    cfg_b.freeze()
    model_b = build_model(cfg_b)
    logs_b = []
    for inp in inputs:
        model_b([inp])
        logs_b.append(json.loads(open(model_b._frame_log_path).read().strip().splitlines()[-1]))
    state_b = [p.detach().clone() for p in model_b.student.parameters()] + [p.detach().clone() for p in model_b.teacher.parameters()]
    n_diff = sum(0 if torch.equal(a, b) else 1 for a, b in zip(state_a, state_b))
    skip = ("frame_ms",)
    fields = [k for k in logs[0] if not k.startswith("audit_") and k not in skip]
    log_equal = all(ra[k] == rb[k] for ra, rb in zip(logs, logs_b) for k in fields)
    print(f"audit on/off: {n_diff} of {len(state_a)} student+teacher tensors differ bitwise; non-audit log fields equal: {log_equal}")
    check("audit on/off: student and teacher states bitwise identical", n_diff == 0)
    check("audit on/off: all non-audit logged quantities identical", log_equal)
    check("audit off writes no audit fields", not any(k.startswith("audit_") for r in logs_b for k in r))
    del model_b, state_b

    # ---- guards ------------------------------------------------------------- #
    model.params[0].grad = torch.ones_like(model.params[0])
    try:
        model([inputs[0]])
        check("stale-gradient guard fires", False)
    except AssertionError:
        check("stale-gradient guard fires", True)
    model.params[0].grad = None
    try:
        model([inputs[0], inputs[1]])
        check("batch size != 1 rejected", False)
    except ValueError:
        check("batch size != 1 rejected", True)
    with torch.no_grad():
        model.params[-1].fill_(float("nan"))
    try:
        model([inputs[0]])
        check("non-finite aborts the run", False)
    except DATDivergenceError:
        check("non-finite aborts the run (no recovery)", os.path.exists(os.path.join(args.out, "dat_bf_DIVERGED.json")))

    print("\nRESULT:", "ALL PASS" if not FAILS else f"FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
