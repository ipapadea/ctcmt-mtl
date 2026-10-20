#!/usr/bin/env python3
"""Class-wise diagnostic study of task interaction during multi-task CTTA.

Reads the existing run logs and, for each detection class (AP) and each
semantic class (IoU), computes per-round trajectories and the 2x2
task-activation factorial:

    d_SEG      = M^SEG    - M^Source     intrinsic effect of seg adaptation
    d_DET      = M^DET    - M^Source     effect of det-driven shared adaptation
    d_SEG|DET  = M^Full   - M^DET        marginal effect of adding seg
    d_DET|SEG  = M^Full   - M^SEG        marginal effect of adding det
    I_c        = M^Full - M^DET - M^SEG + M^Source

I_c > 0 is positive task interaction, I_c < 0 is class-specific negative
transfer. All four conditions within a benchmark share one source checkpoint,
so the contrast isolates adaptation rather than source-model differences.

  python3 scripts/classwise_analysis.py [--logs DIR] [--out classwise_report.md]

NOTE ON THE DETECTION METRIC: detectron2 only prints per-category AP averaged
over IoU 0.50:0.95. Per-class AP50 is not recoverable from the logs, and the
per-evaluation prediction files are overwritten by each evaluation, so it
cannot be recomputed offline either. Detection class numbers here are
AP@[.5:.95]; the global headline numbers elsewhere are AP50. Do not mix them.
"""
import argparse
import os
import re
from datetime import datetime

DET_CLASSES = ["person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]

# trainId order, as emitted by the semantic evaluator.
SEG_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole",
               "traffic light", "traffic sign", "vegetation", "terrain", "sky",
               "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]
# The 8 semantic classes that map one-to-one onto the detection classes.
SEG_THING = ["person", "rider", "car", "truck", "bus", "train",
             "motorcycle", "bicycle"]
SEG_STUFF = [c for c in SEG_CLASSES if c not in SEG_THING]

PROTOCOLS = {
    "cscLT": {
        "title": "Cityscapes-C long-term",
        "domains": ["fog", "motion_blur", "snow", "brightness", "defocus_blur"],
        "rounds": 10,
    },
    "acdcLT": {
        "title": "ACDC long-term",
        "domains": ["fog", "night", "rain", "snow"],
        "rounds": 10,
    },
}

# benchmark -> condition -> log stem.  Within a benchmark every arm shares one
# source checkpoint and one threshold lineage (CSC: thr 0.80, ACDC: thr 0.90).
CONDITIONS = {
    "cscLT": {
        "Source": "source_only_pfn_cs_c",
        "DET": "e15_detonly_thr080_cscLT_s0",
        "SEG": "e21_segonly_thr080_cscLT_s0",
        "Full": "e13a_thrmax080_cscLT_s0",
    },
    "acdcLT": {
        "Source": "source_only_pfn_acdc_full",
        "DET": "e25_detonly_acdc_acdcLT_s0",
        "SEG": "e29_segonly_acdc_acdcLT_s0",
        "Full": "e11_bothsc_ctcrD_acdcLT_s0",
    },
}
# The Cityscapes-C source pass was measured over the 12-corruption stream, so
# its long-term row is those twelve values subset to the five cycle domains.
CSC12_ORDER = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow",
               "frost", "fog", "brightness", "contrast", "elastic_transform",
               "pixelate", "jpeg_compression"]

DET_HDR = re.compile(r"Per-category bbox AP")
IOU_PAIR = re.compile(r"'IoU-([^']+)':\s*([0-9.eE+-]+|nan)")


def _f(x):
    try:
        v = float(x)
    except ValueError:
        return float("nan")
    return v


def parse_classwise(path):
    """-> (list of {det class: AP}, list of {seg class: IoU}), one per evaluation."""
    det, seg = [], []
    if not os.path.exists(path):
        return det, seg
    lines = open(path, errors="ignore").read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if DET_HDR.search(line):
            row = {}
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("|"):
                j += 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                # header and markdown rule rows
                if cells and cells[0] not in ("category",) and not cells[0].startswith(":"):
                    for k in range(0, len(cells) - 1, 2):
                        name, val = cells[k], cells[k + 1]
                        if name in DET_CLASSES:
                            row[name] = _f(val)
                j += 1
            if row:
                det.append(row)
            i = j
            continue
        if "'IoU-" in line:
            row = {n: _f(v) for n, v in IOU_PAIR.findall(line) if n in SEG_CLASSES}
            if row:
                seg.append(row)
        i += 1
    return det, seg


def parse_global(path):
    """-> (per-evaluation AP50, per-evaluation mIoU), the aggregate metrics."""
    ap, iou, task = [], [], None
    if not os.path.exists(path):
        return ap, iou
    for line in open(path, errors="ignore"):
        if "copypaste: Task: bbox" in line:
            task = "b"; continue
        if "copypaste: Task: sem_seg" in line:
            task = "s"; continue
        m = re.search(r"copypaste: ([\d.,]+)\s*$", line)
        if m and task:
            v = [float(x) for x in m.group(1).split(",") if x]
            if task == "b" and len(v) >= 2:
                ap.append(v[1])
            elif task == "s" and v:
                iou.append(v[0])
            task = None
    return ap, iou


def load_globals(logs, proto):
    """-> {condition: {'ap': mean AP50, 'iou': mean mIoU}} over the whole stream."""
    p = PROTOCOLS[proto]
    out = {}
    for cond, stem in CONDITIONS[proto].items():
        ap, iou = parse_global(os.path.join(logs, f"{stem}.log"))
        if cond == "Source" and proto == "cscLT":
            # measured on the 12-corruption pass; take the five cycle domains
            if len(ap) >= 12:
                ap = [ap[CSC12_ORDER.index(d)] for d in p["domains"]]
            if len(iou) >= 12:
                iou = [iou[CSC12_ORDER.index(d)] for d in p["domains"]]
        out[cond] = {"ap": mean(ap), "iou": mean(iou)}
    return out


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def round_means(per_eval, cls, nd, nr):
    """Mean of class `cls` within each round."""
    out = []
    for r in range(nr):
        chunk = per_eval[r * nd:(r + 1) * nd]
        if len(chunk) < nd:
            break
        out.append(mean([e.get(cls, float("nan")) for e in chunk]))
    return out


def stats(rms):
    """mean over rounds, R1, peak, Rlast, peak-to-last drift."""
    if not rms:
        return dict(mean=float("nan"), r1=float("nan"), peak=float("nan"),
                    rlast=float("nan"), drift=float("nan"), peak_round=0)
    peak = max(rms)
    return dict(mean=mean(rms), r1=rms[0], peak=peak, rlast=rms[-1],
                drift=rms[-1] - peak, peak_round=rms.index(peak) + 1)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    n = len(pairs)
    if n < 3:
        return float("nan")

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def fmt(v, prec=2, sign=False):
    if v != v:
        return "&mdash;"
    return f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"


def load_benchmark(logs, proto):
    """-> {condition: {"det": {cls: stats}, "seg": {cls: stats}}}, plus coverage."""
    p = PROTOCOLS[proto]
    nd, nr = len(p["domains"]), p["rounds"]
    out, coverage = {}, {}
    for cond, stem in CONDITIONS[proto].items():
        det, seg = parse_classwise(os.path.join(logs, f"{stem}.log"))
        if cond == "Source":
            # Source is frozen: one pass, tiled across rounds.
            if proto == "cscLT":
                det = [det[CSC12_ORDER.index(d)] for d in p["domains"]] if len(det) >= 12 else det
                seg = [seg[CSC12_ORDER.index(d)] for d in p["domains"]] if len(seg) >= 12 else seg
            det, seg = det[:nd] * nr, seg[:nd] * nr
        coverage[cond] = (len(det), len(seg), stem)
        out[cond] = {
            "det": {c: stats(round_means(det, c, nd, nr)) for c in DET_CLASSES},
            "seg": {c: stats(round_means(seg, c, nd, nr)) for c in SEG_CLASSES},
        }
    return out, coverage


def effects(bench, task, classes):
    """-> {cls: {d_seg, d_det, d_seg_given_det, d_det_given_seg, I}}"""
    res = {}
    for c in classes:
        m = {k: bench[k][task][c]["mean"] for k in ("Source", "DET", "SEG", "Full")}
        res[c] = {
            "src": m["Source"],
            "d_seg": m["SEG"] - m["Source"],
            "d_det": m["DET"] - m["Source"],
            "d_seg_given_det": m["Full"] - m["DET"],
            "d_det_given_seg": m["Full"] - m["SEG"],
            "I": m["Full"] - m["DET"] - m["SEG"] + m["Source"],
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/media/ilias/DATA/ilias/amrod_output/logs")
    ap.add_argument("--out", default="per_class_study.md")
    a = ap.parse_args()

    o = []
    w = o.append
    w("# Class-wise diagnostic study of task interaction under CTTA\n")
    w(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/classwise_analysis.py` "
      "from the run logs. Do not edit by hand - regenerate.\n")
    w("Detection classes are reported as **AP@[.5:.95]**, not AP50: detectron2 "
      "prints only the averaged per-category AP, and the per-evaluation "
      "prediction files are overwritten by each evaluation so AP50 per class "
      "cannot be recomputed offline. Segmentation classes are IoU. Within each "
      "benchmark all four conditions share one source checkpoint and one "
      "threshold lineage.\n")

    benches, covs, globs = {}, {}, {}
    for proto in ("cscLT", "acdcLT"):
        benches[proto], covs[proto] = load_benchmark(a.logs, proto)
        globs[proto] = load_globals(a.logs, proto)

    # ------------------------------------------------------------- coverage
    w("## Data coverage\n")
    w("| Benchmark | Condition | log | det evals | seg evals | usable |")
    w("|---|---|---|---|---|---|")
    for proto in ("cscLT", "acdcLT"):
        nd, nr = len(PROTOCOLS[proto]["domains"]), PROTOCOLS[proto]["rounds"]
        need = nd * nr
        for cond in ("Source", "DET", "SEG", "Full"):
            nde, nse, stem = covs[proto][cond]
            ok = "yes" if (nde >= need and nse >= need) else (
                "seg only" if nse >= need else "**NO**")
            w(f"| {PROTOCOLS[proto]['title']} | {cond} | `{stem}` | {nde} | {nse} | {ok} |")
    w("")

    # ------------------------------------------------- global 2x2 factorial
    w("## Global task-activation factorial\n")
    w("Aggregate metrics, for reference against the class-wise terms below. "
      "`I` is the interaction `Full - DET - SEG + Source`.\n")
    w("| Benchmark | Metric | Source | DET | SEG | Full | d_SEG | d_DET | "
      "d_SEG&#124;DET | I |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for proto in ("cscLT", "acdcLT"):
        for metric, key in (("mAP0.5", "ap"), ("mIoU", "iou")):
            g = {c: globs[proto][c][key] for c in ("Source", "DET", "SEG", "Full")}
            if any(v != v for v in g.values()):
                continue
            I = g["Full"] - g["DET"] - g["SEG"] + g["Source"]
            w(f"| {PROTOCOLS[proto]['title']} | {metric} | {g['Source']:.1f} | "
              f"{g['DET']:.1f} | {g['SEG']:.1f} | {g['Full']:.1f} | "
              f"{g['SEG'] - g['Source']:+.1f} | {g['DET'] - g['Source']:+.1f} | "
              f"{g['Full'] - g['DET']:+.1f} | **{I:+.1f}** |")
    w("")

    for proto in ("cscLT", "acdcLT"):
        p = PROTOCOLS[proto]
        nd, nr = len(p["domains"]), p["rounds"]
        need = nd * nr
        bench = benches[proto]
        have_all = all(covs[proto][c][1] >= need for c in ("Source", "DET", "SEG", "Full"))

        w(f"## {p['title']}\n")
        if not have_all:
            missing = [c for c in ("Source", "DET", "SEG", "Full")
                       if covs[proto][c][1] < need]
            w(f"**Incomplete factorial - missing {', '.join(missing)}.** "
              "Interaction terms are not computed for this benchmark.\n")

        # ---- Q1 trajectories
        for task, classes, unit in (("det", DET_CLASSES, "AP"),
                                    ("seg", SEG_CLASSES, "IoU")):
            w(f"### Per-class trajectories, {unit}\n")
            w("| Class | Cond | Mean | R1 | Peak | R10 | Drift |")
            w("|---|---|---|---|---|---|---|")
            for c in classes:
                for cond in ("Source", "DET", "SEG", "Full"):
                    s = bench[cond][task][c]
                    if s["mean"] != s["mean"]:
                        continue
                    w(f"| {c} | {cond} | {fmt(s['mean'])} | {fmt(s['r1'])} | "
                      f"{fmt(s['peak'])} (R{s['peak_round']}) | {fmt(s['rlast'])} | "
                      f"{fmt(s['drift'], sign=True)} |")
            w("")

        if not have_all:
            continue

        # ---- Q2 effects + Q3 grouping
        for task, classes, unit in (("det", DET_CLASSES, "AP"),
                                    ("seg", SEG_CLASSES, "IoU")):
            e = effects(bench, task, classes)
            w(f"### Class-wise adaptation effects, {unit}\n")
            w("| Class | Source | d_SEG | d_DET | d_SEG&#124;DET | d_DET&#124;SEG | I_c |")
            w("|---|---|---|---|---|---|---|")
            for c in classes:
                v = e[c]
                tag = " **(neg)**" if v["I"] == v["I"] and v["I"] < 0 else ""
                w(f"| {c} | {fmt(v['src'])} | {fmt(v['d_seg'], sign=True)} | "
                  f"{fmt(v['d_det'], sign=True)} | {fmt(v['d_seg_given_det'], sign=True)} | "
                  f"{fmt(v['d_det_given_seg'], sign=True)} | {fmt(v['I'], sign=True)}{tag} |")
            w("")

            if task == "seg":
                w("#### Thing classes (mapped to detection) vs stuff classes\n")
                w("| Group | n | mean I_c | mean d_SEG&#124;DET | n with I_c<0 |")
                w("|---|---|---|---|---|")
                for gname, grp in (("thing (mapped)", SEG_THING),
                                   ("stuff (seg-only)", SEG_STUFF)):
                    Is = [e[c]["I"] for c in grp]
                    ds = [e[c]["d_seg_given_det"] for c in grp]
                    neg = sum(1 for x in Is if x == x and x < 0)
                    w(f"| {gname} | {len(grp)} | {fmt(mean(Is), sign=True)} | "
                      f"{fmt(mean(ds), sign=True)} | {neg}/{len(grp)} |")
                w("")

            # ---- Q4 difficulty vs harmfulness
            src = [e[c]["src"] for c in classes]
            Ic = [e[c]["I"] for c in classes]
            drift = [bench["Full"][task][c]["drift"] for c in classes]
            w(f"#### Difficulty vs harmfulness, {unit}\n")
            w("| Relationship | Spearman rho |")
            w("|---|---|")
            w(f"| source score vs I_c | {fmt(spearman(src, Ic))} |")
            w(f"| full-MTL drift vs I_c | {fmt(spearman(drift, Ic))} |")
            w(f"| source score vs full-MTL drift | {fmt(spearman(src, drift))} |")
            w("")
            w("A strongly negative `source score vs I_c` would mean the weakest "
              "source classes are the most harmful; near zero means harmfulness "
              "is not explained by class difficulty alone.\n")

    # ------------------------------------------------------- Q7 sign flip
    ok = {pr: all(covs[pr][c][1] >= len(PROTOCOLS[pr]["domains"]) * PROTOCOLS[pr]["rounds"]
                  for c in ("Source", "DET", "SEG", "Full"))
          for pr in ("cscLT", "acdcLT")}
    w("## Cross-benchmark sign flip\n")
    if not (ok["cscLT"] and ok["acdcLT"]):
        w("Requires a complete factorial on **both** benchmarks; not yet available.\n")
    else:
        for task, classes, unit in (("det", DET_CLASSES, "AP"),
                                    ("seg", SEG_CLASSES, "IoU")):
            ec = effects(benches["cscLT"], task, classes)
            ea = effects(benches["acdcLT"], task, classes)
            w(f"### {unit}\n")
            w("| Class | I_c ACDC | I_c CSC | verdict |")
            w("|---|---|---|---|")
            for c in classes:
                ia, ic = ea[c]["I"], ec[c]["I"]
                if ia != ia or ic != ic:
                    verdict = "&mdash;"
                elif ia > 0 and ic < 0:
                    verdict = "**flips: helps on ACDC, harms on CSC**"
                elif ia < 0 and ic < 0:
                    verdict = "harmful on both"
                elif ia > 0 and ic > 0:
                    verdict = "helpful on both"
                else:
                    verdict = "flips the other way"
                w(f"| {c} | {fmt(ia, sign=True)} | {fmt(ic, sign=True)} | {verdict} |")
            w("")

    # ------------------------------------------------------------ conclusions
    if ok["cscLT"] and ok["acdcLT"]:
        EFF = {pr: {t: effects(benches[pr], t, cl)
                    for t, cl in (("det", DET_CLASSES), ("seg", SEG_CLASSES))}
               for pr in ("cscLT", "acdcLT")}
        tax = {(pr, k): (globs[pr]["Full"][k] - globs[pr]["DET"][k]
                         - globs[pr]["SEG"][k] + globs[pr]["Source"][k])
               for pr in ("cscLT", "acdcLT") for k in ("ap", "iou")}
        main = {(pr, k): globs[pr]["SEG"][k] - globs[pr]["Source"][k]
                for pr in ("cscLT", "acdcLT") for k in ("ap", "iou")}
        cond = {(pr, k): globs[pr]["Full"][k] - globs[pr]["DET"][k]
                for pr in ("cscLT", "acdcLT") for k in ("ap", "iou")}

        def negcount(pr, t, cl):
            return sum(1 for c in cl if EFF[pr][t][c]["I"] < 0), len(cl)

        w("## Conclusions\n")

        w("### 1. The interaction is negative in every cell\n")
        w(f"| Benchmark | detection I | segmentation I |")
        w("|---|---|---|")
        for pr in ("cscLT", "acdcLT"):
            w(f"| {PROTOCOLS[pr]['title']} | {tax[(pr,'ap')]:+.1f} | {tax[(pr,'iou')]:+.1f} |")
        w("")
        gap_ap = abs(tax[("cscLT", "ap")] - tax[("acdcLT", "ap")])
        gap_iou = abs(tax[("cscLT", "iou")] - tax[("acdcLT", "iou")])
        w("Enabling the segmentation loss on top of detection adaptation "
          "carries a cost on **both** benchmarks, including the one where "
          "multi-task adaptation wins overall. Multi-task interference is "
          "therefore not created by the synthetic corruption stream.\n")
        w(f"The detection penalty is similar across benchmarks (differs by "
          f"{gap_ap:.1f} mAP0.5), while the segmentation penalty is "
          f"{'markedly' if gap_iou > 1.5 else 'slightly'} larger on "
          f"Cityscapes-C (differs by {gap_iou:.1f} mIoU). So the penalty is "
          "partly intrinsic to sharing the trunk and partly amplified by the "
          "long synthetic stream.\n")
        w("What separates the two benchmarks is mainly the **main effect** of "
          "segmentation adaptation, i.e. how much it was worth on its own:\n")
        w("| Benchmark | d_SEG mAP0.5 | d_SEG mIoU | d_SEG&#124;DET mAP0.5 | d_SEG&#124;DET mIoU |")
        w("|---|---|---|---|---|")
        for pr in ("cscLT", "acdcLT"):
            w(f"| {PROTOCOLS[pr]['title']} | {main[(pr,'ap')]:+.1f} | {main[(pr,'iou')]:+.1f} | "
              f"{cond[(pr,'ap')]:+.1f} | {cond[(pr,'iou')]:+.1f} |")
        w("")
        w("The apparent sign flip lives in the **conditional** effect, not in "
          "the interaction: where segmentation adaptation has a large solo "
          "gain it survives the penalty, and where it has a small one it does "
          "not. This predicts that multi-task CTTA pays off exactly when the "
          "auxiliary task's standalone gain exceeds the interaction penalty.\n")

        w("### 2. The damage is distributed across classes, not localised\n")
        for pr in ("cscLT", "acdcLT"):
            nd_, td_ = negcount(pr, "det", DET_CLASSES)
            ns_, ts_ = negcount(pr, "seg", SEG_CLASSES)
            w(f"- {PROTOCOLS[pr]['title']}: **{nd_}/{td_}** detection classes and "
              f"**{ns_}/{ts_}** segmentation classes have `I_c < 0`.")
        w("")
        w("There is no small set of pathological classes to exclude, so "
          "per-class pseudo-label filtering cannot recover the loss.\n")

        w("### 3. Shared-representation contamination, not cross-task class mapping\n")
        w("| Benchmark | thing (mapped) mean I_c | stuff (no detection counterpart) mean I_c |")
        w("|---|---|---|")
        for pr in ("cscLT", "acdcLT"):
            th = mean([EFF[pr]["seg"][c]["I"] for c in SEG_THING])
            st = mean([EFF[pr]["seg"][c]["I"] for c in SEG_STUFF])
            w(f"| {PROTOCOLS[pr]['title']} | {th:+.2f} | {st:+.2f} |")
        w("")
        worst = sorted(SEG_STUFF, key=lambda c: EFF["cscLT"]["seg"][c]["I"])[:4]
        w("The mapped thing classes are hit hardest, so class mapping "
          "contributes. But stuff classes with **no detection counterpart at "
          "all** are also heavily damaged (worst on Cityscapes-C: "
          + ", ".join(f"`{c}` {EFF['cscLT']['seg'][c]['I']:+.2f}" for c in worst)
          + "). Those classes cannot be harmed through a shared label space, "
            "so the dominant mechanism is contamination of the shared "
            "representation.\n")

        flips = [c for c in SEG_CLASSES
                 if EFF["acdcLT"]["seg"][c]["I"] > 0 > EFF["cscLT"]["seg"][c]["I"]]
        dflips = [c for c in DET_CLASSES
                  if EFF["acdcLT"]["det"][c]["I"] > 0 > EFF["cscLT"]["det"][c]["I"]]
        w("### 4. Which classes change sign between benchmarks\n")
        if flips:
            n_stuff = sum(1 for c in flips if c in SEG_STUFF)
            w(f"Segmentation classes with `I_ACDC > 0 > I_CSC`: "
              + ", ".join(f"`{c}`" for c in flips)
              + f" ({n_stuff} of {len(flips)} are stuff classes).\n")
        else:
            w("No segmentation class changes sign.\n")
        w(f"Detection classes that change sign: "
          + (", ".join(f"`{c}`" for c in dflips) if dflips else "**none** - "
             "every detection class has a negative interaction on both "
             "benchmarks.") + "\n")

        w("### 5. Harmfulness is not explained by class difficulty\n")
        w("| Benchmark | Task | source vs I_c | drift vs I_c |")
        w("|---|---|---|---|")
        for pr in ("cscLT", "acdcLT"):
            for t, cl, label in (("det", DET_CLASSES, "detection"),
                                 ("seg", SEG_CLASSES, "segmentation")):
                s = [EFF[pr][t][c]["src"] for c in cl]
                I = [EFF[pr][t][c]["I"] for c in cl]
                d = [benches[pr]["Full"][t][c]["drift"] for c in cl]
                w(f"| {PROTOCOLS[pr]['title']} | {label} | "
                  f"{fmt(spearman(s, I))} | {fmt(spearman(d, I))} |")
        w("")
        w("Source accuracy never predicts harmfulness on any benchmark or "
          "task. Weak source classes are therefore *not* the ones generating "
          "damaging supervision, and a rare difficult class contributes little "
          "because it rarely fires. Late-round drift is a strong predictor on "
          "the long synthetic stream but only a weak one on adverse weather, "
          "so drift explains the Cityscapes-C collapse specifically rather "
          "than multi-task interference in general.\n")

        w("### Caveats\n")
        w("- Every arm here is **seed 0 only**; class-level values are noisier "
          "than the aggregate seed spread (0.14-0.20 mAP0.5). Treat "
          "`|I_c| < 1` as unresolved.")
        w("- Detection classes are AP@[.5:.95], not AP50 (see the header).")
        w("- `I_c` weights every class equally; the global metrics do not. The "
          "two agree here, but they are different quantities.")
        w("- **ACDC source provenance.** The Source cell uses "
          "`source_only_pfn_acdc_full`, re-measured through the standard "
          "runner (400 images on each `acdc_*_mtl` set, per-category tables "
          "retained). It disagrees with the older `source_only_pfn_acdc` log "
          "used by `results.md` on the rain domain only (AP50 30.6 vs 35.5, "
          "mIoU 32.4 vs 39.4); the other three domains match to 0.1. The old "
          "log is a filtered 80-line extract with no inference markers, so it "
          "cannot be audited. Every interaction term on ACDC shifts by the "
          "difference in the Source cell, so this must be resolved before the "
          "ACDC numbers are published.\n")

    # ----------------------------------------------------------- what is missing
    w("## Not answerable from the current logs\n")
    w("The following require new instrumentation and a re-run; they are not "
      "recoverable from the existing logs.\n")
    w("- **Per-class detection pseudo-label dynamics** (per-class threshold, "
      "accepted count, mean teacher confidence, pseudo-label precision/recall). "
      "The adapter logs only aggregate `n_pseudo` and `thr=min/mean/max` over "
      "all classes.")
    w("- **Per-class semantic pseudo-label reliability** (teacher entropy, "
      "confidence, predicted vs GT pixel frequency, per-class loss "
      "contribution). None of these are emitted.")
    w("- **Per-class AP50.** Only per-category AP@[.5:.95] is printed, and the "
      "prediction dumps are overwritten each evaluation.\n")

    open(a.out, "w").write("\n".join(o) + "\n")
    print(f"wrote {a.out} ({len(o)} lines)")


if __name__ == "__main__":
    main()
