"""Source-only eval on a single Cityscapes-C corruption in AMROD's D2 pipeline.

Loads a plain GeneralizedRCNN (NOT the AMROD meta-arch, which self-adapts even
in eval mode), applies AMROD's cfg_cityscapes_c_short.yaml, and evaluates COCO
mAP on a chosen corruption. Writes per-corruption results to
{out_dir}/{corruption}/results.json.

Env var CITYSCAPES_C_CORRUPTION selects the corruption; DETECTRON2_DATASETS
points at the cityscapes-c root that AMROD's builtin registration uses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corruption", default=None,
                    help="override CITYSCAPES_C_CORRUPTION env var")
    args = ap.parse_args()

    corruption = args.corruption or os.environ.get("CITYSCAPES_C_CORRUPTION")
    if not corruption:
        raise SystemExit("must pass --corruption or set CITYSCAPES_C_CORRUPTION")

    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.modeling import build_model
    from detectron2.utils.logger import setup_logger

    setup_logger(name="d2_srconly")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # Force plain FR-CNN (the AMROD meta-arch runs forward_and_adapt even in eval mode).
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.DATASETS.TEST = (corruption,)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = 1

    out_root = Path(args.out_dir) / corruption
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(out_root)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    loader = build_detection_test_loader(cfg, corruption)
    # output_dir=None suppresses COCOEvaluator writing instances_predictions.pth
    # (can be ~200 MB per corruption). We only need the aggregate mAP number.
    evaluator = COCOEvaluator(corruption, output_dir=None)
    results = inference_on_dataset(model, loader, evaluator)

    bbox = results.get("bbox", {}) if isinstance(results, dict) else {}
    summary = {
        "corruption": corruption,
        "weights": args.weights,
        "mAP": round(float(bbox.get("AP", 0.0)) / 100.0, 4),
        "mAP_50": round(float(bbox.get("AP50", 0.0)) / 100.0, 4),
        "mAP_75": round(float(bbox.get("AP75", 0.0)) / 100.0, 4),
        "raw_bbox": bbox,
    }
    with open(out_root / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[d2_srconly] {corruption}: mAP={summary['mAP']:.4f} mAP@50={summary['mAP_50']:.4f}")


if __name__ == "__main__":
    main()
