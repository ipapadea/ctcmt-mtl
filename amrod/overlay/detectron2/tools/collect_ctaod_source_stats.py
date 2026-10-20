#!/usr/bin/env python
"""
Offline CTAOD source-feature statistics (Yoo et al., CVPR'24).

Reproduces `TEST.COLLECT_FEATURES` of the official repository
(detectron2/engine/defaults.py L638-655 + GeneralizedRCNN.collect_feats):

  gl[k]      : (mean, cov) of the globally average-pooled FPN feature of level k
  fg[c]      : (mean, cov) of the box-head features of proposals assigned to
               class c by IoU>=0.5 with a GT box (c = num_classes -> background)
  kl_div[k]  : in-domain symmetric KL between the full-sample Gaussian and the
               first-20-sample Gaussian of level k (used only by CTAOD-Skip)

Run ONCE on the source training split with the canonical source checkpoint.
The resulting file is the only source-derived asset used at test time; no
source image is ever accessed during online adaptation.

Example:
  python detectron2/tools/collect_ctaod_source_stats.py \
      --config-file detectron2/configs/Cityscapes/newsrc_ctaod_acdc.yaml \
      --dataset cityscapes_fine_instance_seg_train \
      --num-images 2000 --seed 0 \
      --output /workspace/output/ctaod_stats/newsrc_ctaod_source_stats.pt
"""
import argparse
import logging
import os
import random
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.run_provenance import log_run_provenance

logger = logging.getLogger("ctaod.collect")


class _GaussianAccum:
    """Streaming mean/covariance in float64 (identical to torch.cov)."""

    def __init__(self, dim, device):
        self.n = 0
        self.s1 = torch.zeros(dim, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def update(self, feats):
        f = feats.double()
        self.n += f.shape[0]
        self.s1 += f.sum(dim=0)
        self.s2 += f.t() @ f

    def finalize(self):
        if self.n < 2:
            return None
        mean = self.s1 / self.n
        cov = (self.s2 - self.n * torch.outer(mean, mean)) / (self.n - 1)
        cov = 0.5 * (cov + cov.t())
        return mean.float().cpu(), cov.float().cpu(), self.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--dataset", default="cityscapes_fine_instance_seg_train")
    ap.add_argument("--num-images", type=int, default=2000,
                    help="official CTAOD uses 2000 source images; -1 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--max-per-class-per-image", type=int, default=100)
    ap.add_argument("--output", required=True)
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    setup_logger(name="ctaod.collect")
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list([o for o in args.opts if o != "--"])
    cfg.freeze()

    log_run_provenance(
        cfg,
        method="CTAOD-source-statistics-collection",
        extra={
            "collect_dataset": args.dataset,
            "collect_num_images": args.num_images,
            "collect_seed": args.seed,
            "collect_iou_thr": args.iou_thr,
            "collect_max_per_class_per_image": args.max_per_class_per_image,
            "collect_output": args.output,
        },
        logger=logger,
    )

    device = torch.device(cfg.MODEL.DEVICE)
    model = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    num_classes = model.roi_heads.num_classes

    dicts = DatasetCatalog.get(args.dataset)
    dicts = sorted(dicts, key=lambda d: d["file_name"])
    if args.num_images > 0 and args.num_images < len(dicts):
        rng = random.Random(args.seed)
        dicts = sorted(rng.sample(dicts, args.num_images), key=lambda d: d["file_name"])
    logger.info("collecting statistics from {} images of {}".format(len(dicts), args.dataset))

    # Test-time preprocessing (identical to adaptation), but annotations kept.
    mapper = DatasetMapper(
        is_train=True,
        augmentations=[T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)],
        image_format=cfg.INPUT.FORMAT,
        use_instance_mask=False,
        recompute_boxes=False,
    )
    loader = build_detection_test_loader(dicts, mapper=mapper, num_workers=4)

    torch.manual_seed(args.seed)
    gl_feats = {}
    fg_acc = {}
    roi = model.roi_heads

    with torch.no_grad():
        for i, inputs in enumerate(loader):
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            for k in features:
                gl_feats.setdefault(k, []).append(features[k].mean(dim=[2, 3]).cpu())

            gt_instances = [x["instances"].to(device) for x in inputs]
            proposals, _ = model.proposal_generator(images, features, gt_instances)

            box_feats_in = [features[f] for f in roi.box_in_features]
            box_features = roi.box_pooler(box_feats_in, [x.proposal_boxes for x in proposals])
            box_features = roi.box_head(box_features)

            iou_with_gt = torch.empty(0, device=device)
            gt_label = torch.empty(0, device=device)
            for j, p in enumerate(proposals):
                if len(gt_instances[j]) > 0:
                    _iou, _idx = pairwise_iou(p.proposal_boxes, gt_instances[j].gt_boxes).max(dim=1)
                    _lab = torch.where(
                        _iou > args.iou_thr,
                        gt_instances[j].gt_classes[_idx],
                        torch.tensor(num_classes, device=device),
                    )
                else:
                    _iou = torch.zeros(len(p), device=device)
                    _lab = torch.full((len(p),), num_classes, device=device)
                iou_with_gt = torch.cat([iou_with_gt, _iou])
                gt_label = torch.cat([gt_label, _lab.float()])
            gt_label = gt_label.long()

            for _c in torch.unique(gt_label):
                c = int(_c.item())
                sel = box_features[gt_label == c]
                pick = torch.randperm(sel.shape[0])[: args.max_per_class_per_image]
                sel = sel[pick]
                if c not in fg_acc:
                    fg_acc[c] = _GaussianAccum(sel.shape[1], device)
                fg_acc[c].update(sel)

            if (i + 1) % 200 == 0:
                logger.info("{}/{} images".format(i + 1, len(dicts)))

    feat_stats = {"gl": {}, "fg": {}, "kl_div": {}}
    meta = {"fg_counts": {}, "gl_counts": {}}
    for k, v in gl_feats.items():
        f = torch.cat(v, dim=0)
        mean, cov = f.mean(dim=0), torch.cov(f.T)
        feat_stats["gl"][k] = (mean, cov)
        meta["gl_counts"][k] = int(f.shape[0])
        d1 = torch.distributions.MultivariateNormal(mean.double(), cov.double())
        d2 = torch.distributions.MultivariateNormal(f[:20, :].mean(dim=0).double(), cov.double())
        kl = (
            torch.distributions.kl.kl_divergence(d1, d2)
            + torch.distributions.kl.kl_divergence(d1, d2)
        ) / 2
        feat_stats["kl_div"][k] = kl.float().cpu()
        logger.info("gl[{}] n={} in-domain KL={:.4f}".format(k, f.shape[0], kl.item()))

    for c in sorted(fg_acc):
        out = fg_acc[c].finalize()
        if out is None:
            logger.warning("class {} has <2 samples; skipped".format(c))
            continue
        mean, cov, n = out
        feat_stats["fg"][c] = (mean, cov)
        meta["fg_counts"][c] = n
        logger.info(
            "fg[{}] n={} |mean|={:.4f} cov.max={:.4f}".format(
                c, n, mean.norm().item(), cov.max().item()
            )
        )
        if n < 1000:
            logger.warning(
                "class {}: only {} samples for a {}-dim covariance".format(c, n, mean.shape[0])
            )

    feat_stats["meta"] = {
        "dataset": args.dataset,
        "num_images": len(dicts),
        "seed": args.seed,
        "iou_thr": args.iou_thr,
        "max_per_class_per_image": args.max_per_class_per_image,
        "model_weights": cfg.MODEL.WEIGHTS,
        "num_classes": num_classes,
        **meta,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(feat_stats, args.output)
    logger.info("saved {}".format(args.output))


if __name__ == "__main__":
    main()
