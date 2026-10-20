"""Byte-equivalent reproduction of AMROD's corrupt.py inside pinned env.

Runs cv2.imread (BGR) -> imagecorruptions.corrupt() -> cv2.imwrite, matching
AMROD/corrupt.py exactly. imagecorruptions==1.1.2 is pinned in the image.

Adds:
  - skimage compat shim: imagecorruptions 1.1.2 passes multichannel=True to
    skimage.filters.gaussian which was removed in skimage>=0.20. We monkey-
    patch it to translate to channel_axis=-1. Visual output is identical to
    old skimage behaviour.
  - parallel via ProcessPoolExecutor
  - resumable (skips existing files)
  - per-image np.random.seed (hashed from filename) for determinism across
    restarts; pass --no-seed for AMROD's original undefined-random behaviour.

Layout matches D2's register_cityscapes_c expectation:
  {dst_root}/{corruption}/leftImg8bit/val/{city}/{basename}
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# 12 Cityscapes-C corruptions in AMROD Table 2 order (skips gaussian_noise,
# shot_noise, impulse_noise per AMROD's b filter in corrupt.py).
AMROD_CORRUPTIONS = [
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
SEVERITY = 5


def _install_skimage_compat_shims():
    try:
        import skimage.filters as _sf
    except Exception:
        return
    _orig = getattr(_sf, "_gaussian_original", None) or _sf.gaussian

    def _gaussian_shim(*args, **kwargs):
        if "multichannel" in kwargs:
            mc = kwargs.pop("multichannel")
            if mc and "channel_axis" not in kwargs:
                kwargs["channel_axis"] = -1
        return _orig(*args, **kwargs)

    _sf._gaussian_original = _orig
    _sf.gaussian = _gaussian_shim


def _seed_for(name: str) -> int:
    h = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def _process_one(args_tuple):
    src_path, dst_path, corruption, seed = args_tuple
    _install_skimage_compat_shims()
    from imagecorruptions import corrupt

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return dst_path, "skip"

    img = cv2.imread(src_path, cv2.IMREAD_COLOR)  # BGR, same as AMROD
    if img is None:
        return dst_path, "read_fail"

    if seed is not None:
        np.random.seed(seed)

    out = corrupt(img, corruption_name=corruption, severity=SEVERITY)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ok = cv2.imwrite(dst_path, out)
    return dst_path, ("ok" if ok else "write_fail")


def _collect_jobs(src_root: Path, dst_root: Path, corruption: str, use_seed: bool):
    src_val = src_root / "leftImg8bit" / "val"
    if not src_val.is_dir():
        raise SystemExit(f"missing {src_val}; expected clean cityscapes leftImg8bit/val/")

    dst_val = dst_root / corruption / "leftImg8bit" / "val"
    jobs = []
    for city_dir in sorted(src_val.iterdir()):
        if not city_dir.is_dir():
            continue
        for src_path in sorted(city_dir.glob("*.png")):
            rel = src_path.relative_to(src_val)
            dst_path = dst_val / rel
            seed = _seed_for(f"{corruption}/{rel.as_posix()}") if use_seed else None
            jobs.append((str(src_path), str(dst_path), corruption, seed))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--corruption", default=None,
                    help="single corruption to run, else all 12 AMROD corruptions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-seed", action="store_true")
    args = ap.parse_args()

    use_seed = not (args.no_seed or os.environ.get("AMROD_NO_SEED"))

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    corruptions = [args.corruption] if args.corruption else AMROD_CORRUPTIONS
    for c in corruptions:
        if c not in AMROD_CORRUPTIONS:
            raise SystemExit(f"unknown corruption {c!r}")

    for c in corruptions:
        jobs = _collect_jobs(src_root, dst_root, c, use_seed)
        n = len(jobs)
        print(f"[corrgen] {c}: {n} images -> {dst_root}/{c}/", flush=True)
        done, skipped, failed = 0, 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, fut in enumerate(as_completed(ex.submit(_process_one, j) for j in jobs), 1):
                _, status = fut.result()
                if status == "ok":
                    done += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                if i % 50 == 0 or i == n:
                    print(f"[corrgen] {c}: {i}/{n} (new={done} skip={skipped} fail={failed})", flush=True)
        if failed:
            print(f"[corrgen] {c}: WARNING {failed} failures", file=sys.stderr, flush=True)

    print("[corrgen] done", flush=True)


if __name__ == "__main__":
    main()
