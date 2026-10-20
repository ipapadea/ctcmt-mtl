"""Offline semantic scorer for DAT-BF saved predictions.

Scores the saved *primary* (gated, pre-update EMA teacher) and *secondary*
(pre-update plain scale-1.0 teacher) label maps written by DATPFN under
``<run_dir>/dat_bf_predictions/{primary,plain_view}/<frame>.png`` with the stock
``SemSegEvaluator`` (same classes, ignore label and metric code as the online
evaluator). Frames are mapped to datasets sequentially, in stream order.

Partial scoring: only frames that exist (and are logged in dat_bf_frames.jsonl)
are scored, e.g. after a numerical divergence; ``complete`` is reported per dataset.

  python tools/dat_bf_score_predictions.py --run-dir RUN --datasets defocus_blur_semseg:100 \
      [--online-json RUN/online_results.json] [--online-log LOG]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

from detectron2.data import DatasetCatalog
from detectron2.evaluation import SemSegEvaluator

TAGS = ("primary", "plain_view")
METRIC_KEYS = ("mIoU", "fwIoU", "mACC", "pACC")


class _LabelMap:
    """Stands in for the model's (C,H,W) score tensor: the evaluator only calls argmax(dim=0)."""

    def __init__(self, labels):
        self._labels = torch.from_numpy(labels.astype(np.int64))

    def argmax(self, dim=0):
        return self._labels


def parse_specs(specs):
    out = []
    for s in specs:
        name, _, n = s.partition(":")
        out.append((name, int(n) if n else None))
    return out


def read_frame_log(run_dir):
    path = os.path.join(run_dir, "dat_bf_frames.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def spec_keys(specs):
    """Result keys: the dataset name when it occurs once in the stream, else ``NNN_<name>`` (stream
    position, so repeated domains in long-term streams never overwrite each other)."""
    counts = {}
    for name, _ in specs:
        counts[name] = counts.get(name, 0) + 1
    return [name if counts[name] == 1 else f"{i:03d}_{name}" for i, (name, _) in enumerate(specs)]


def score_run(run_dir, specs, tags=TAGS):
    log = read_frame_log(run_dir)
    results = {tag: {} for tag in tags}
    offset = 0
    keys = spec_keys(specs)
    cache = {}
    for key, (name, limit) in zip(keys, specs):
        if name not in cache:
            cache[name] = DatasetCatalog.get(name)
        records = cache[name]
        if limit is not None:
            records = records[:limit]
        n_avail = max(0, min(len(records), len(log) - offset))
        for tag in tags:
            pred_dir = os.path.join(run_dir, "dat_bf_predictions", tag)
            ev = SemSegEvaluator(name, distributed=False, output_dir=None)
            ev.reset()
            scored = 0
            for i in range(n_avail):
                frame = offset + i + 1
                rec = records[i]
                if log[offset + i]["frame"] != frame or log[offset + i]["file_name"] != rec["file_name"]:
                    raise RuntimeError(
                        f"frame/dataset misalignment at frame {frame}: log={log[offset + i]['file_name']} "
                        f"dataset={rec['file_name']}"
                    )
                png = os.path.join(pred_dir, f"{frame:06d}.png")
                if not os.path.exists(png):
                    break
                ev.process([{"file_name": rec["file_name"]}], [{"sem_seg": _LabelMap(np.array(Image.open(png)))}])
                scored += 1
            res = ev.evaluate()["sem_seg"] if scored else {}
            results[tag][key] = {
                "name": name,
                "frames_scored": scored,
                "frames_expected": len(records),
                "complete": scored == len(records),
                **{k: float(res[k]) for k in METRIC_KEYS if k in res},
            }
        offset += len(records)
    return results


def compare_online(offline_primary, online):
    """``online`` maps dataset -> {"sem_seg": {...}} or dataset -> {...}. Returns max abs diff."""
    worst = 0.0
    detail = {}
    for name, off in offline_primary.items():
        # truncated smoke datasets are evaluated online as "<name>_first<N>"
        on = online.get(name) or online.get(f"{name}_first{off['frames_expected']}")
        if on is None and "name" in off:
            on = online.get(f"{off['name']}_first{off['frames_expected']}")
        if on is None:
            continue
        on = on.get("sem_seg", on)
        diffs = {k: abs(float(on[k]) - off[k]) for k in METRIC_KEYS if k in on and k in off}
        detail[name] = diffs
        worst = max([worst] + list(diffs.values()))
    return worst, detail


def parse_online_log_list(path):
    """Ordered [(dataset, metrics)] (4 decimals) from the 'copypaste' lines of a run log, one entry per
    evaluation in stream order (repeated datasets are kept)."""
    text = open(path, errors="ignore").read()
    out = []
    pattern = re.compile(
        r"Evaluation results for (\S+) in csv format:.*?copypaste: Task: sem_seg\s*\n.*?"
        r"copypaste: mIoU,fwIoU,mACC,pACC\s*\n.*?copypaste: ([0-9.,\-naninf]+)",
        re.S,
    )
    for m in pattern.finditer(text):
        vals = [float(v) for v in m.group(2).split(",")]
        out.append((m.group(1), dict(zip(METRIC_KEYS, vals))))
    return out


def parse_online_log(path):
    """Per-dataset online metrics; for repeated datasets the keys follow :func:`spec_keys`."""
    ordered = parse_online_log_list(path)
    counts = {}
    for name, _ in ordered:
        counts[name] = counts.get(name, 0) + 1
    return {(name if counts[name] == 1 else f"{i:03d}_{name}"): vals for i, (name, vals) in enumerate(ordered)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset[:max_frames] in stream order")
    ap.add_argument("--online-json")
    ap.add_argument("--online-log")
    ap.add_argument("--tol", type=float, default=1e-9, help="tolerance for --online-json")
    ap.add_argument("--log-tol", type=float, default=6e-5, help="tolerance for --online-log (4 printed decimals)")
    args = ap.parse_args()

    out_path = os.path.join(args.run_dir, "dat_bf_offline_scores.json")
    if os.path.exists(out_path):
        sys.exit(f"refusing to overwrite {out_path}")
    results = score_run(args.run_dir, parse_specs(args.datasets))
    report = {"scores": results}
    status = 0
    for src, loader, tol in (("online_json", lambda p: json.load(open(p)), args.tol),
                             ("online_log", parse_online_log, args.log_tol)):
        path = getattr(args, src)
        if not path:
            continue
        worst, detail = compare_online(results["primary"], loader(path))
        ok = worst <= tol and len(detail) == len(results["primary"])  # every offline entry must be matched
        report[f"parity_{src}"] = {"max_abs_diff": worst, "tolerance": tol, "agree": ok, "detail": detail}
        status |= 0 if ok else 1
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    for tag in TAGS:
        for name, r in results[tag].items():
            print(f"[{tag:10s}] {name}: " + " ".join(f"{k}={r[k]:.4f}" for k in METRIC_KEYS if k in r)
                  + f" frames={r['frames_scored']}/{r['frames_expected']}")
    for key in ("parity_online_json", "parity_online_log"):
        if key in report:
            print(key, "max_abs_diff=%.3e" % report[key]["max_abs_diff"], "AGREE" if report[key]["agree"] else "DISAGREE")
    sys.exit(status)


if __name__ == "__main__":
    main()
