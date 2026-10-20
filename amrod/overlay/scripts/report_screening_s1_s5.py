#!/usr/bin/env python3
"""Report the S1..S5 negative-transfer screening batch against its controls.

Reads the Cityscapes-C long-term logs directly and prints, per arm:
  - final-round and mean AP50 / mIoU, and the per-round AP50 trajectory
    (the plateau shows up as a flat tail from round ~5)
  - the gradient-conflict diagnostics the mechanism produced

Interpretation anchors on this stream, same source checkpoint, seed 0:
  e13a full MTL thr 0.8   AP50 24.80  <- negative-transfer reference
  e15  det-only thr 0.8   AP50 26.80  <- detection control
  AMROD on PFN            AP50 26.26

  python3 scripts/report_screening_s1_s5.py [--logs DIR]
"""
import argparse
import os
import re
from collections import defaultdict

BBOX_RE = re.compile(r"copypaste: ([\d.,]+)\s*$")
GRAD_RE = re.compile(r"\[CT-CMT-GRAD\] iter=(\d+) mode=(\w+) (.*)")
KV_RE = re.compile(r"(\w+)=(-?[\d.nan]+)")
DIAG_RE = re.compile(r"\[CT-CMT-GRADDIAG\] iter=(\d+) (.*)")

ARMS = [
    ("e16_protectedgrad_cscLT_s0", "S1 protected grad projection"),
    ("e17_cagrad_cscLT_s0",        "S2 CAGrad consensus"),
    ("e18_harddecouple_cscLT_s0",  "S3 hard decoupling"),
    ("e19_frozentrunk_cscLT_s0",   "S4 frozen shared trunk"),
    ("e20_dynweight_cscLT_s0",     "S5 dynamic aux weight"),
    ("e22_seghead_only_cscLT_s0",  "S6 seg-head-only routing"),
    ("e23_detseg_nocross_cscLT_s0", "C1 det+seg, no cross-task"),
]
CONTROLS = [
    ("e13a_thrmax080_cscLT_s0", "B0 e13a full MTL"),
    ("e15_detonly_thr080_cscLT_s0", "B1 e15 det-only"),
    ("e21_segonly_thr080_cscLT_s0", "B2 e21 seg-only"),
    ("amrod_pfnsrc_cscLT_cronus_s0", "AMROD (PFN src)"),
]

# ACDC uses a 4-domain cycle and a different reference recipe (E11, thr 0.9).
ACDC_CONTROLS = [
    ("e11_bothsc_ctcrD_acdcLT_s0", "B0 e11 full MTL"),
    ("e25_detonly_acdc_acdcLT_s0", "B1 e25 det-only"),
    ("amrod_pfnsrc_acdcLT_s0", "AMROD (PFN src)"),
    ("tent_pfnsrc_acdcLT_s0", "TENT (PFN src)"),
    ("cotta_pfnsrc_acdcLT_s0", "CoTTA (PFN src)"),
]
ACDC_ARMS = [
    ("e24_seghead_only_acdc_acdcLT_s0", "S6 seg-head-only routing"),
]

PROTOCOLS = {
    "csc":  {"domains": 5, "rounds": 10, "controls": CONTROLS, "arms": ARMS,
             "ref": ("e13a", 24.80), "ceiling": ("e15", 26.80)},
    "acdc": {"domains": 4, "rounds": 10, "controls": ACDC_CONTROLS, "arms": ACDC_ARMS,
             "ref": ("e11", None), "ceiling": ("e25", None)},
}


def parse_log(path):
    """Per-evaluation AP50 / mIoU, plus the last gradient-diagnostic lines."""
    ap50, miou, grad, comps = [], [], [], []
    crashed = False
    task = None
    for line in open(path, errors="ignore"):
        if "Traceback (most recent call last)" in line:
            crashed = True
        if "copypaste: Task: bbox" in line:
            task = "bbox"
            continue
        if "copypaste: Task: sem_seg" in line:
            task = "sem_seg"
            continue
        if "copypaste: AP," in line or "copypaste: mIoU," in line:
            continue
        m = BBOX_RE.search(line)
        if m and task:
            vals = [float(x) for x in m.group(1).split(",") if x]
            if task == "bbox" and len(vals) >= 2:
                ap50.append(vals[1])
            elif task == "sem_seg" and vals:
                miou.append(vals[0])
            task = None
            continue
        m = GRAD_RE.search(line)
        if m:
            kv = {k: float(v) for k, v in KV_RE.findall(m.group(3))}
            kv["iter"] = int(m.group(1))
            kv["mode"] = m.group(2)
            grad.append(kv)
            continue
        m = DIAG_RE.search(line)
        if m:
            comps.append({k: float(v) for k, v in KV_RE.findall(m.group(2))})
    return ap50, miou, grad, comps, crashed


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    """Sample standard deviation; nan for fewer than two seeds."""
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def rounds(vals, n=10, complete_only=True, per=5):
    """Mean per round of ``per`` domains.

    A partial round is not comparable: the cycle mixes an easy domain (fog,
    AP50 ~47) with a catastrophic one (snow, AP50 ~1), so averaging 2 of 5
    domains reads as a spurious jump on an in-progress run.
    """
    out = []
    for i in range(n):
        chunk = vals[i * per:(i + 1) * per]
        if not chunk or (complete_only and len(chunk) < per):
            continue
        out.append(mean(chunk))
    return out


def seed_table(entries, logdir, seeds, per, nr):
    """Aggregate each arm over its seed variants: mean +- std of the run means."""
    total = per * nr
    print(f"\nSEED-AGGREGATED  (seeds {', '.join(map(str, seeds))})")
    print(f"{'arm':<34}{'n':>3}{'AP50 mean+-std':>20}{'mIoU mean+-std':>20}   per-seed AP50")
    print("-" * 112)
    for name, label in entries:
        base = name.rsplit("_s", 1)[0]
        aps, ious, tags = [], [], []
        for s in seeds:
            path = os.path.join(logdir, f"{base}_s{s}.log")
            if not os.path.exists(path):
                continue
            ap50, miou, _, _, crashed = parse_log(path)
            if crashed or max(len(ap50), len(miou)) != total:
                tags.append(f"s{s}:incomplete")
                continue
            if ap50:
                aps.append(mean(ap50))
            if miou:
                ious.append(mean(miou))
            tags.append(f"s{s}:{mean(ap50):.2f}" if ap50 else f"s{s}:--")
        if not tags:
            print(f"{label:<34}{'--':>3}   (no complete runs)")
            continue
        a = f"{mean(aps):.2f} +- {std(aps):.2f}" if aps else "--"
        i = f"{mean(ious):.2f} +- {std(ious):.2f}" if ious else "--"
        print(f"{label:<34}{len(aps) or len(ious):>3}{a:>20}{i:>20}   {' '.join(tags)}")
    print("\nA difference smaller than the spread between seeds is not a result.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="/media/ilias/DATA/ilias/amrod_output/logs")
    p.add_argument("--protocol", default="csc", choices=sorted(PROTOCOLS))
    p.add_argument("--seeds", default="",
                   help="comma-separated seeds; aggregates each arm across them")
    a = p.parse_args()
    proto = PROTOCOLS[a.protocol]
    per, nr = proto["domains"], proto["rounds"]
    total = per * nr
    entries = proto["controls"] + proto["arms"]

    if a.seeds:
        seed_table(entries, a.logs, [int(s) for s in a.seeds.split(",")], per, nr)
        return

    # The two anchors are read from the runs themselves, so the table stays
    # correct when a protocol's baselines are re-measured.
    def _mean_ap_iou(stem):
        path = os.path.join(a.logs, f"{stem}.log")
        if not os.path.exists(path):
            return None, None
        ap, iou, _, _, _ = parse_log(path)
        k = len(rounds(ap, nr, per=per)) * per
        return (mean(ap[:k]) if k else None), (mean(iou[:k]) if k else None)

    ref_ap, _ = _mean_ap_iou(proto["controls"][0][0])
    ceil_ap, _ = _mean_ap_iou(proto["controls"][1][0])
    ref_name = proto["ref"][0]
    ceil_name = proto["ceiling"][0]

    print(f"protocol: {a.protocol}  ({per} domains x {nr} rounds = {total} evaluations)")
    print(f"{'arm':<34}{'n':>4}{'AP50 mean':>11}{'AP50 R%d' % nr:>10}"
          f"{'mIoU mean':>11}{'mIoU R%d' % nr:>10}   vs {ref_name} / vs {ceil_name}")
    print("-" * 112)
    rows = {}
    for name, label in entries:
        path = os.path.join(a.logs, f"{name}.log")
        if not os.path.exists(path):
            print(f"{label:<34}{'--':>4}   (missing {name}.log)")
            continue
        ap50, miou, grad, comps, crashed = parse_log(path)
        rows[name] = (label, ap50, miou, grad, comps)
        r_ap = rounds(ap50, nr, per=per)
        r_iou = rounds(miou, nr, per=per)
        # Seg-only baselines emit no AP50 and det-only ones no mIoU; judge
        # completeness on whichever metric the arm actually produces.
        k = max(len(r_ap), len(r_iou)) * per
        n_evals = max(len(ap50), len(miou))
        m_ap = mean(ap50[:k])
        d13 = m_ap - ref_ap if ref_ap is not None else float("nan")
        d15 = m_ap - ceil_ap if ceil_ap is not None else float("nan")
        if crashed:
            flag = f"  *** CRASHED/STALE, {k // per} rounds - DO NOT COMPARE ***"
        elif n_evals == total:
            flag = ""
        else:
            flag = f"  (running, {k // per} full rounds)"
        print(f"{label:<34}{n_evals:>4}{m_ap:>11.2f}"
              f"{(r_ap[-1] if r_ap else float('nan')):>10.2f}"
              f"{mean(miou[:k]):>11.2f}{(r_iou[-1] if r_iou else float('nan')):>10.2f}"
              f"   {d13:+6.2f} / {d15:+6.2f}{flag}")

    print(f"\nAP50 per round ({per} domains each) -- the plateau is a flat tail:")
    for name, (label, ap50, miou, _, _) in rows.items():
        traj = " ".join(f"{v:5.1f}" for v in rounds(ap50, nr, per=per))
        print(f"  {label:<32} {traj}")
    print("\nmIoU per round:")
    for name, (label, ap50, miou, _, _) in rows.items():
        traj = " ".join(f"{v:5.1f}" for v in rounds(miou, nr, per=per))
        print(f"  {label:<32} {traj}")

    print("\nGradient-conflict diagnostics (cumulative at end of stream):")
    for name, (label, _, _, grad, comps) in rows.items():
        if not grad:
            print(f"  {label:<32} (none -- no shared-trunk surgery in this arm)")
            continue
        g = grad[-1]
        print(f"  {label:<32} mode={g['mode']} cos_mean={g.get('cos_mean', float('nan')):+.4f} "
              f"conflict_rate={g.get('conflict_rate', float('nan')):.3f} "
              f"applied_rate={g.get('applied_rate', float('nan')):.3f} "
              f"|g_det|={g.get('g_det', float('nan')):.3f} "
              f"|g_aux|={g.get('g_aux', float('nan')):.3f} "
              f"gamma_mean={g.get('gamma_mean', float('nan')):.3f} "
              f"w_det_mean={g.get('w_det_mean', float('nan')):.3f} "
              f"steps={int(g.get('n_steps', 0))}")

    print("\nPer-component conflict with detection (mean over the stream):")
    print("  which non-detection loss actually fights the detector, and where")
    keys = ("cos_det_seg", "cos_det_ctcl", "cos_det_ctcr", "cos_det_proto",
            "blk_res2", "blk_res3", "blk_res4", "blk_res5", "blk_fpn")
    for name, (label, _, _, _, comps) in rows.items():
        if not comps:
            continue
        agg = defaultdict(list)
        for c in comps:
            for k, v in c.items():
                if k != "iter" and v == v:
                    agg[k].append(v)
        cells = " ".join(f"{k.replace('cos_det_', '').replace('blk_', ''):>6}="
                         f"{mean(agg[k]):+.3f}" for k in keys if k in agg)
        print(f"  {label:<32} {cells}  (n={len(comps)})")


if __name__ == "__main__":
    main()
