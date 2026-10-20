"""Rewrite a Cityscapes COCO-style JSON so that each `file_name` points at the
Foggy-Cityscapes rendering at a given fog density.

Cityscapes' foggy split (Sakaridis et al., 2018) keeps the exact same labels
as the underlying clear Cityscapes val set; only the RGB filenames change:

    clear : <city>/<basename>_leftImg8bit.png
    foggy : <city>/<basename>_leftImg8bit_foggy_beta_{beta}.png

Given the mmdet-converter output for clear Cityscapes val
(``instancesonly_filtered_gtFine_val.json``) this script writes one JSON per
beta with the file_names patched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _rewrite(clear_json_path: Path, beta: str, out_path: Path) -> int:
    with open(clear_json_path) as f:
        data = json.load(f)
    suffix_new = f"_leftImg8bit_foggy_beta_{beta}.png"
    n = 0
    for img in data.get("images", []):
        fn = img["file_name"]
        if fn.endswith("_leftImg8bit.png"):
            img["file_name"] = fn[: -len("_leftImg8bit.png")] + suffix_new
            n += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clear-json", required=True,
                   help="Path to instancesonly_filtered_gtFine_val.json")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write per-beta jsons into.")
    p.add_argument("--betas", nargs="+", default=["0.005", "0.01", "0.02"],
                   help="Fog densities to emit jsons for.")
    args = p.parse_args()

    src = Path(args.clear_json)
    out_dir = Path(args.out_dir)
    for beta in args.betas:
        out = out_dir / f"instancesonly_filtered_gtFine_val_foggy_beta_{beta}.json"
        n = _rewrite(src, beta, out)
        print(f"[foggy] beta={beta:<5} images_patched={n:<5} -> {out}")


if __name__ == "__main__":
    main()
