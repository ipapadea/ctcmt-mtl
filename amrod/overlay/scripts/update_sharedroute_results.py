#!/usr/bin/env python3
"""Refresh the adaptive shared-route section of results.md from the run logs.

Only fully finished runs (50 evals for cscLT, 40 for acdcLT) are included, so it
can be re-run as additional seeds complete. The section lives between two marker
comments; everything else in results.md is left untouched.

  python3 scripts/update_sharedroute_results.py
"""
import glob
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gather_results import parse_log  # noqa: E402

RESULTS = os.path.join(os.path.dirname(HERE), "results.md")
LOGS = os.environ.get("HOST_OUT", "/media/ilias/DATA/ilias/amrod_output") + "/logs"
BEGIN, END = "<!-- sharedroute:begin -->", "<!-- sharedroute:end -->"

PROTOS = {
    "cscLT": dict(title="Cityscapes-C long-term", n=50,
                  doms=["Fog", "Motion", "Snow", "Bright", "Defocus"]),
    "acdcLT": dict(title="ACDC long-term", n=40, doms=["Fog", "Night", "Rain", "Snow"]),
}
ROUNDS = (1, 5, 10)


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    return m, s


def load(proto):
    runs = {}
    for p in glob.glob(f"{LOGS}/adaptive_sharedroute_{proto}_s*.log"):
        seed = int(re.search(r"_s(\d+)\.log$", p).group(1))
        _, ev = parse_log(p)
        if len(ev) == PROTOS[proto]["n"] and all(e["AP50"] is not None and e["mIoU"] is not None for e in ev):
            runs[seed] = ev
    return dict(sorted(runs.items()))


def per_domain_table(runs, k, metric):
    nd = len(PROTOS[k]["doms"])
    head = [f"R{r} {d}" for r in ROUNDS for d in PROTOS[k]["doms"]]
    out = ["| Seed | " + " | ".join(head) + " | Mean |", "|---|" + "---|" * (len(head) + 1)]
    for seed, ev in runs.items():
        vals = [ev[(r - 1) * nd + j][metric] for r in ROUNDS for j in range(nd)]
        allv = [e[metric] for e in ev]
        out.append(f"| {seed} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | **{sum(allv)/len(allv):.1f}** |")
    return out


def section():
    L = ["## Adaptive shared-route (new-source, long-term)", "",
         "Config `newsrc_adaptive_sharedroute_{cscLT,acdcLT}.yaml`, Panoptic FPN R50 MTL source, no reset, "
         "online batch size 1. Maintained by `scripts/update_sharedroute_results.py` (re-run it as seeds finish); "
         "only completed runs are listed.", ""]
    summ = ["| Protocol | Seeds | AP50 | mIoU | R1 AP50 | R10 AP50 | R1 mIoU | R10 mIoU |",
            "|---|---|---|---|---|---|---|---|"]
    detail = []
    for k, cfg in PROTOS.items():
        runs = load(k)
        if not runs:
            continue
        n = len(cfg["doms"])
        cols = {"AP50": [], "mIoU": [], "R1 AP50": [], "R10 AP50": [], "R1 mIoU": [], "R10 mIoU": []}
        for ev in runs.values():
            ap, iu = [e["AP50"] for e in ev], [e["mIoU"] for e in ev]
            for name, v in (("AP50", ap), ("mIoU", iu)):
                cols[name].append(sum(v) / len(v))
            cols["R1 AP50"].append(sum(ap[:n]) / n)
            cols["R10 AP50"].append(sum(ap[-n:]) / n)
            cols["R1 mIoU"].append(sum(iu[:n]) / n)
            cols["R10 mIoU"].append(sum(iu[-n:]) / n)

        def cell(v):
            m, s = mean_std(v)
            return f"{m:.2f}" + (f" &plusmn; {s:.2f}" if s is not None else "")

        summ.append(f"| `{k}` | {','.join(map(str, runs))} | " + " | ".join(cell(cols[c]) for c in cols) + " |")
        detail += [f"### {cfg['title']} &mdash; mAP0.5 (rounds 1, 5, 10)", ""] + per_domain_table(runs, k, "AP50")
        detail += ["", f"### {cfg['title']} &mdash; mIoU (rounds 1, 5, 10)", ""] + per_domain_table(runs, k, "mIoU") + [""]
    if len(summ) == 2:
        return None
    return "\n".join(L + summ + [""] + detail).rstrip() + "\n"


def main():
    body = section()
    if body is None:
        sys.exit("no completed shared-route runs found")
    block = f"{BEGIN}\n{body}{END}\n"
    txt = open(RESULTS).read()
    if BEGIN in txt:
        txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", lambda _: block, txt, flags=re.S)
    else:
        anchor = "## Reproducibility and caveats"
        i = txt.index(anchor)
        txt = txt[:i] + block + "\n" + txt[i:]
    open(RESULTS, "w").write(txt)
    print(body)


if __name__ == "__main__":
    main()
