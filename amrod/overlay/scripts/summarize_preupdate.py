#!/usr/bin/env python3
"""Pre-update vs post-update teacher prediction (CTCMT and Base, ACDC4 and CSC12, 3 seeds).

Reads the evaluation logs of
  post-update : headline_lam1_*, paper_base_*             (adapt-then-predict, reported runs)
  pre-update  : paper26_preupd_*                          (SOLVER.CTCMT_PREDICT_BEFORE_UPDATE=True)
and writes per_run.csv, per_domain.csv and summary.md into --out.

Per run, AP50 / mIoU are averaged over the domains of the stream, as in the slide tables.
The std in summary.md is the sample std (n-1) over seeds.

    python3 scripts/summarize_preupdate.py [--logs DIR] [--meta DIR] [--out DIR]
"""
import argparse
import csv
import os
import shutil
import statistics as st

SEEDS = (0, 42, 123)
CASES = [
    # protocol, method, post-update log, pre-update log, expected domains
    ("ACDC4", "CTCMT", "headline_lam1_acdc4_s{s}", "paper26_preupd_acdc4_ctcmt_s{s}", 4),
    ("ACDC4", "Base", "paper_base_acdc4_s{s}", "paper26_preupd_acdc4_base_s{s}", 4),
    ("CSC12", "CTCMT", "headline_lam1_csc12_s{s}", "paper26_preupd_csc12_ctcmt_s{s}", 12),
    ("CSC12", "Base", "paper_base_csc12_s{s}", "paper26_preupd_csc12_base_s{s}", 12),
]


def parse(path):
    """Per-domain (AP50, mIoU) lists in stream order."""
    lines = open(path, errors="ignore").read().splitlines()
    ap, miou = [], []
    for i, line in enumerate(lines):
        if "copypaste: Task: bbox" in line:
            ap.append(float(lines[i + 2].split("copypaste: ")[1].split(",")[1]))
        if "copypaste: mIoU,fwIoU" in line:
            miou.append(float(lines[i + 1].split("copypaste: ")[1].split(",")[0]))
    return ap, miou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="/media/ilias/DATA/ilias/amrod_output/logs")
    ap.add_argument("--meta", default="/media/ilias/DATA/ilias/amrod_output/paper26_meta")
    ap.add_argument("--out", default="/media/ilias/DATA/ilias/AMROD/results/preupdate_prediction")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per_run, per_domain, runs = [], [], {}
    for proto, method, post_t, pre_t, n in CASES:
        for mode, tmpl in (("post", post_t), ("pre", pre_t)):
            for s in SEEDS:
                name = tmpl.format(s=s)
                a, m = parse(os.path.join(args.logs, name + ".log"))
                if len(a) != n or len(m) != n:
                    raise SystemExit(f"{name}: expected {n} evals, found AP50={len(a)} mIoU={len(m)}")
                for d, (x, y) in enumerate(zip(a, m)):
                    per_domain.append([proto, method, mode, s, d, x, y, name])
                per_run.append([proto, method, mode, s, st.mean(a), st.mean(m), name])
                runs[(proto, method, mode, s)] = (st.mean(a), st.mean(m))

    with open(os.path.join(args.out, "per_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "method", "prediction", "seed", "AP50_mean_over_domains", "mIoU_mean_over_domains", "run"])
        w.writerows([r[:4] + [f"{r[4]:.4f}", f"{r[5]:.4f}", r[6]] for r in per_run])
    with open(os.path.join(args.out, "per_domain.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "method", "prediction", "seed", "domain_index", "AP50", "mIoU", "run"])
        w.writerows([r[:5] + [f"{r[5]:.4f}", f"{r[6]:.4f}", r[7]] for r in per_domain])

    def ms(v):
        return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"

    def col(proto, method, mode, k):
        return [runs[(proto, method, mode, s)][k] for s in SEEDS]

    L = ["# Pre-update vs post-update teacher prediction", "",
         "CTCMT and Base, 3 seeds (0, 42, 123). Mean ± sample std over seeds of the per-run domain mean.",
         "post = adapt-then-predict (reported runs); pre = predict-then-adapt (`SOLVER.CTCMT_PREDICT_BEFORE_UPDATE=True`).", "",
         "| Protocol | Method | Post AP50 | Pre AP50 | Δ AP50 (pre−post) | Post mIoU | Pre mIoU | Δ mIoU (pre−post) |",
         "|---|---|---|---|---|---|---|---|"]
    for proto, method, *_ in CASES:
        d = [st.mean([b - a for a, b in zip(col(proto, method, "post", k), col(proto, method, "pre", k))]) for k in (0, 1)]
        L.append(f"| {proto} | {method} | {ms(col(proto, method, 'post', 0))} | {ms(col(proto, method, 'pre', 0))} | {d[0]:+.3f} "
                 f"| {ms(col(proto, method, 'post', 1))} | {ms(col(proto, method, 'pre', 1))} | {d[1]:+.3f} |")
    L += ["", "## Per-seed Δ (pre − post)", "", "| Protocol | Method | Δ AP50 per seed | Δ mIoU per seed |", "|---|---|---|---|"]
    for proto, method, *_ in CASES:
        dA = [col(proto, method, "pre", 0)[i] - col(proto, method, "post", 0)[i] for i in range(3)]
        dM = [col(proto, method, "pre", 1)[i] - col(proto, method, "post", 1)[i] for i in range(3)]
        L.append(f"| {proto} | {method} | {', '.join(f'{x:+.2f}' for x in dA)} | {', '.join(f'{x:+.2f}' for x in dM)} |")
    L += ["", "## CTCMT − Base", "", "| Protocol | Prediction | Δ AP50 | Δ mIoU |", "|---|---|---|---|"]
    for proto in ("ACDC4", "CSC12"):
        for mode in ("post", "pre"):
            a = st.mean(col(proto, "CTCMT", mode, 0)) - st.mean(col(proto, "Base", mode, 0))
            m = st.mean(col(proto, "CTCMT", mode, 1)) - st.mean(col(proto, "Base", mode, 1))
            L.append(f"| {proto} | {mode} | {a:+.3f} | {m:+.3f} |")
    L += ["", "## Provenance", "",
          "- Pre-update runs: `scripts/paper26_batch_preupdate.sh`; run metadata copied to `meta/`.",
          "- The flag was added to `ctcmt_mtl.py` / `defaults.py` and was uncommitted when these runs were executed",
          "  (`git_dirty` in the meta files); record the commit that contains it before citing the numbers.",
          "- Short-term protocols only (ACDC4, CSC12); no long-term pre-update runs."]
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(L) + "\n")

    meta_out = os.path.join(args.out, "meta")
    os.makedirs(meta_out, exist_ok=True)
    for fn in sorted(os.listdir(args.meta)):
        if fn.startswith("paper26_preupd_"):
            shutil.copy(os.path.join(args.meta, fn), meta_out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
