#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from statistics import mean

CYCLE = ["fog", "motion_blur", "snow", "brightness", "defocus_blur"]

def floats_after_key(text, key):
    # Handles dict-like lines such as 'AP50': 37.2 or "mIoU": 42.1
    pat = re.compile(rf"""['"]?{re.escape(key)}['"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)""")
    return [float(m.group(1)) for m in pat.finditer(text)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    ap50 = floats_after_key(text, "AP50")
    miou = floats_after_key(text, "mIoU")

    lines = []
    e = lines.append
    e("=" * 96)
    e("CT-CR MIXED LONG-TERM QUICK SUMMARY")
    e("=" * 96)
    e(f"Log            : {args.log}")
    e(f"AP50 values    : {len(ap50)}")
    e(f"mIoU values    : {len(miou)}")
    e("")

    def section(name, vals):
        e(name)
        e("-" * 96)
        if len(vals) < 50:
            e(f"Expected at least 50 values; found {len(vals)}. Inspect log manually.")
            e("")
            return

        vals = vals[-50:]
        e(f"Overall mean: {mean(vals):.4f}")
        e("")
        e(f"{'Round':>5} " + " ".join(f"{c:>14}" for c in CYCLE) + f" {'round_mean':>14}")
        for r in range(10):
            row = vals[r*5:(r+1)*5]
            e(f"{r+1:>5} " + " ".join(f"{x:>14.4f}" for x in row) + f" {mean(row):>14.4f}")
        e("")
        e("Per-corruption:")
        for j, c in enumerate(CYCLE):
            xs = vals[j::5]
            peak = max(xs)
            peak_r = xs.index(peak) + 1
            e(
                f"  {c:<14} mean={mean(xs):.4f} "
                f"R1={xs[0]:.4f} R10={xs[-1]:.4f} "
                f"R10-R1={xs[-1]-xs[0]:+.4f} "
                f"peak={peak:.4f}@R{peak_r} "
                f"last-peak={xs[-1]-peak:+.4f}"
            )
        e("")

    section("BBOX AP50", ap50)
    section("SEMANTIC mIoU", miou)

    report = "\n".join(lines)
    print(report)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
