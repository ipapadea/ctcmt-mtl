# CTAOD (Yoo et al., CVPR 2024) --- "What, How, and When Should Object
# Detectors Update in Continually Changing Test Domains?"
#
# Same-source port of the official implementation
#   https://github.com/natureyoo/ContinualTTA_ObjectDetection
# onto this repository's canonical Cityscapes Panoptic-FPN R50 checkpoint
# (detection branch only).  The adaptation algorithm is kept as close to the
# official code as possible; only the plumbing differs (the official code puts
# the online loop inside a custom evaluator, here it lives inside the
# meta-architecture so the standard detectron2 eval loop can be reused).
#
# WHAT IS ADAPTED: only the newly inserted, zero-initialised parallel
# bottleneck adapters in the ResNet BottleneckBlocks.  Every source parameter
# stays frozen, and the adapters start as exact no-ops, so the initial
# function of the source checkpoint is unchanged.
#
# This baseline NEVER touches the semantic segmentation branch.
import json
import logging
import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.run_provenance import log_run_provenance

from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN

__all__ = ["CTAOD_Yoo", "CTAODAdapter", "attach_ctaod_adapters"]

logger = logging.getLogger(__name__)


class CTAODAdapter(nn.Module):
    """
    Official `Adapter` (modeling/backbone/resnet.py) : 1x1 down-projection,
    ReLU, 1x1 up-projection.  Up-projection is zero-initialised so the block
    output is unchanged at t=0.
    """

    def __init__(self, in_planes: int, planes: int, r: int = 32):
        super().__init__()
        hidden = max(in_planes // r, 1)
        self.down_proj = nn.Conv2d(in_planes, hidden, kernel_size=1, bias=True)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Conv2d(hidden, planes, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.non_linear_func(self.down_proj(x)))


def _bottleneck_forward_with_adapter(self, x):
    """
    Replica of BottleneckBlock.forward with the official parallel adapter:
        out = conv3(relu(conv2(relu(conv1(x))))) + shortcut + adapter(out_tmp)
    `out_tmp` is the conv2 output after the in-place ReLU, exactly as in the
    official code (`out = F.relu_(out_tmp)` mutates `out_tmp`).
    """
    out = self.conv1(x)
    out = F.relu_(out)

    out_tmp = self.conv2(out)
    out = F.relu_(out_tmp)

    out = self.conv3(out)

    if self.shortcut is not None:
        shortcut = self.shortcut(x)
    else:
        shortcut = x

    out = out + shortcut + self.ctaod_adapter(out_tmp)
    out = F.relu_(out)
    return out


def attach_ctaod_adapters(model: nn.Module, ratio: int = 32):
    """
    Insert one parallel adapter into every BottleneckBlock of the ResNet
    bottom-up trunk, without touching the shared resnet.py implementation
    (the block's `forward` is rebound per instance).
    """
    import types

    device = next(model.parameters()).device
    blocks = []
    for stage in model.backbone.bottom_up.stages:
        for block in stage:
            bottleneck_channels = block.conv2.out_channels
            out_channels = block.conv3.out_channels
            adapter = CTAODAdapter(bottleneck_channels, out_channels, r=ratio).to(device)
            block.add_module("ctaod_adapter", adapter)
            block.forward = types.MethodType(_bottleneck_forward_with_adapter, block)
            blocks.append(block)
    return blocks


@META_ARCH_REGISTRY.register()
class CTAOD_Yoo(nn.Module):
    """
    Online continual test-time adaptation of a frozen GeneralizedRCNN with
    CTAOD adapters + source-feature distribution alignment.
    """

    @configurable
    def __init__(self, *, model: nn.Module, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.model.eval()

        c = cfg.TEST.CTAOD_YOO
        self.alpha_gl = c.ALPHA_GLOBAL
        self.alpha_fg = c.ALPHA_FOREGROUND
        self.gl_align = c.GLOBAL_ALIGN if c.GLOBAL_ALIGN else None
        self.fg_align = c.FOREGROUND_ALIGN if c.FOREGROUND_ALIGN else None
        self.th_bg = c.TH_BG
        self.ema_gamma = c.EMA_GAMMA
        self.freq_weight = c.FREQ_WEIGHT
        self.skip_redundant = c.SKIP_REDUNDANT if c.SKIP_REDUNDANT else None
        self.skip_tau = c.SKIP_TAU
        self.skip_period = c.SKIP_PERIOD
        self.skip_beta = c.SKIP_BETA
        self.num_classes = self.model.roi_heads.num_classes

        # ---- adapters (the only trainable parameters) --------------------
        self.model.requires_grad_(False)
        self.blocks = attach_ctaod_adapters(self.model, ratio=c.ADAPTER_RATIO)
        params = []
        for b in self.blocks:
            b.ctaod_adapter.requires_grad_(True)
            params += list(b.ctaod_adapter.parameters())
        self.adapter_params = params

        lr = cfg.SOLVER.BASE_LR * max(cfg.SOLVER.IMS_PER_BATCH // 4, 1)
        optimizer = torch.optim.SGD(
            [{"params": params}],
            lr,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
        self.optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        # ---- offline source statistics -----------------------------------
        assert os.path.isfile(c.SOURCE_STATS_PATH), (
            "CTAOD source statistics not found: {}".format(c.SOURCE_STATS_PATH)
        )
        self.s_stats = torch.load(c.SOURCE_STATS_PATH, map_location="cpu")
        self.t_stats = {}
        self.template_cov = {}
        self.ema_n = {}
        self._s_dist_cache = {}
        self.initialize()

        # ---- online skip state (official evaluator state) ----------------
        self.cur_used = True
        self.pause_iter = 0
        self.loss_ema99 = 0.0
        self.n_updates = 0
        self.n_steps = 0
        self.div_thr = 2.0 * sum(float(v) for v in self.s_stats["kl_div"].values()) * (
            self.skip_tau if self.skip_redundant is not None else 1.0
        )

        # ---- pre/post-update diagnostic ----------------------------------
        self.diag_frames = int(c.DIAG_PREPOST_FRAMES)
        self.diag_path = os.path.join(cfg.OUTPUT_DIR, "ctaod_prepost_diag.jsonl")
        self.diag_records = []
        self.skip_diag_path = os.path.join(cfg.OUTPUT_DIR, "ctaod_skip_diag.jsonl")
        self.skip_diag_records = []

        n_adapter = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log_run_provenance(
            cfg,
            method="CTAOD-Skip" if self.skip_redundant else "CTAOD",
            extra={
                "ctaod_source_stats": c.SOURCE_STATS_PATH,
                "ctaod_adapter_ratio": c.ADAPTER_RATIO,
                "ctaod_adapters": len(self.blocks),
                "ctaod_adapter_params": n_adapter,
                "trainable_params": n_train,
                "model_params_total": n_total,
                "ctaod_lr": lr,
                "ctaod_skip_redundant": self.skip_redundant,
                "ctaod_div_thr": self.div_thr,
                "ctaod_freq_weight": self.freq_weight,
                "semantic_branch_used": False,
            },
            logger=logger,
        )
        assert n_train == n_adapter, "only adapters may be trainable"

    def _record_skip_diag(self, record):
        self.skip_diag_records.append(record)
        if len(self.skip_diag_records) >= 25:
            with open(self.skip_diag_path, "a") as f:
                for item in self.skip_diag_records:
                    f.write(json.dumps(item) + "\n")
            self.skip_diag_records.clear()

    @classmethod
    def from_config(cls, cfg):
        model = META_ARCH_REGISTRY.get("GeneralizedRCNN")(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        model.to(torch.device(cfg.MODEL.DEVICE))
        return {"model": model, "cfg": cfg}

    @property
    def device(self):
        return self.model.pixel_mean.device

    # ------------------------------------------------------------------ #
    def initialize(self):
        """Official `GeneralizedRCNN.initialize()`."""
        if self.gl_align == "KL":
            self.t_stats["gl"], self.template_cov["gl"] = {}, {}
            for k in self.s_stats["gl"]:
                mean, cov = self.s_stats["gl"][k]
                self.template_cov["gl"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["gl"][k] = (mean, cov)
        if self.fg_align == "KL":
            self.t_stats["fg"], self.template_cov["fg"] = {}, {}
            for k in self.s_stats["fg"]:
                mean, cov = self.s_stats["fg"][k]
                self.template_cov["fg"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["fg"][k] = (mean, cov)
                self.ema_n[k] = 0

    def _source_dist(self, group, k):
        """KL( s || t ) uses a constant source Gaussian; cache it (pure speedup)."""
        key = (group, k)
        if key not in self._s_dist_cache:
            mean = self.s_stats[group][k][0].to(self.device)
            cov = self.s_stats[group][k][1].to(self.device) + self.template_cov[group][k].to(
                self.device
            )
            self._s_dist_cache[key] = (
                torch.distributions.MultivariateNormal(mean, cov),
                cov,
            )
        return self._s_dist_cache[key]

    # ------------------------------------------------------------------ #
    def _forward_detector(self, batched_inputs, need_loss: bool):
        """
        One detector forward.  Returns (outputs, features, box_features,
        class_logits).  Predictions are detached for the evaluator so no graph
        is retained through the reported boxes/masks.
        """
        m = self.model
        images = m.preprocess_image(batched_inputs)
        features = m.backbone(images.tensor)
        proposals, _ = m.proposal_generator(images, features, None)

        roi = m.roi_heads
        box_feats_in = [features[f] for f in roi.box_in_features]
        box_features = roi.box_pooler(box_feats_in, [x.proposal_boxes for x in proposals])
        box_features = roi.box_head(box_features)
        predictions = roi.box_predictor(box_features)

        with torch.no_grad():
            det_predictions = (predictions[0].detach(), predictions[1].detach())
            pred_instances, _ = roi.box_predictor.inference(det_predictions, proposals)
            results = roi.forward_with_given_boxes(features, pred_instances)
            outputs = GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        if not need_loss:
            return outputs, None, None, None
        return outputs, features, box_features, predictions[0]

    def _adapt_losses(self, features, box_features, class_logits):
        """Official `GeneralizedRCNN.adapt()` loss computation."""
        adapt_loss = {}

        if self.fg_align is not None:
            _scores = nn.Softmax(dim=1)(class_logits)
            bg_scores = _scores[:, -1]
            fg_scores, fg_preds = _scores[:, :-1].max(dim=1)

            valid = fg_scores >= self.th_bg
            fg_preds[~valid] = (
                torch.ones((~valid).sum()).long().to(valid.device) * self.num_classes
            )
            fg_scores[~valid] = bg_scores[~valid]

            loss_fg_align = 0
            loss_n = 0
            for _k in fg_preds[fg_preds != self.num_classes].unique():
                k = _k.item()
                if k not in self.t_stats["fg"]:
                    continue
                if (fg_preds == k).sum() > 0:
                    cur_feats = box_features[fg_preds == k]
                    self.ema_n[k] += cur_feats.shape[0]
                    diff = cur_feats - self.t_stats["fg"][k][0][None, :].to(self.device)
                    delta = 1 / self.ema_gamma * diff.sum(dim=0)
                    cur_target_mean = self.t_stats["fg"][k][0].to(self.device) + delta

                    s_dist, cov = self._source_dist("fg", k)
                    t_dist = torch.distributions.MultivariateNormal(cur_target_mean, cov)
                    if self.freq_weight:
                        ema_max = max(
                            [self.ema_n[_c] for _c in range(self.num_classes) if _c in self.ema_n]
                        )
                        class_weight = np.log(max(ema_max, 1) / max(self.ema_n[k], 1))
                        cur_loss = min(class_weight + 0.01, 10) ** 2 * (
                            (
                                torch.distributions.kl.kl_divergence(s_dist, t_dist)
                                + torch.distributions.kl.kl_divergence(t_dist, s_dist)
                            )
                            / 2
                        )
                    else:
                        cur_loss = (
                            torch.distributions.kl.kl_divergence(s_dist, t_dist)
                            + torch.distributions.kl.kl_divergence(t_dist, s_dist)
                        ) / 2
                    if cur_loss < 10 ** 5:
                        loss_fg_align = loss_fg_align + cur_loss
                        self.t_stats["fg"][k] = (cur_target_mean.detach(), None)
                        loss_n += 1
            if loss_n > 0:
                adapt_loss["fg_align"] = self.alpha_fg * loss_fg_align

        if self.gl_align is not None:
            loss_gl_align = 0
            for k in features.keys():
                if k not in self.t_stats["gl"]:
                    continue
                cur_feats = features[k].mean(dim=[2, 3])
                diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
                delta = 1 / self.ema_gamma * diff.sum(dim=0)
                cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta
                s_dist, cov = self._source_dist("gl", k)
                t_dist = torch.distributions.MultivariateNormal(cur_target_mean, cov)
                cur_loss = (
                    torch.distributions.kl.kl_divergence(s_dist, t_dist)
                    + torch.distributions.kl.kl_divergence(t_dist, s_dist)
                ) / 2
                if cur_loss < 10 ** 5:
                    loss_gl_align = loss_gl_align + cur_loss
                    self.t_stats["gl"][k] = (cur_target_mean.detach(), None)
            adapt_loss["global_align"] = self.alpha_gl * loss_gl_align

        return adapt_loss

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        self.n_steps += 1
        idx = self.n_steps - 1

        check = (
            self.skip_redundant is None
            or self.cur_used
            or (not self.cur_used and self.pause_iter % self.skip_period == 0)
            or ("period" in self.skip_redundant and idx % self.skip_period == 0)
        )

        if not check:
            with torch.no_grad():
                outputs, _, _, _ = self._forward_detector(batched_inputs, need_loss=False)
            self.pause_iter += 1
            self._record_skip_diag(
                {
                    "iter": self.n_steps,
                    "idx": idx,
                    "statistics_evaluated": False,
                    "updated": False,
                    "pause_iter": self.pause_iter,
                    "ga": None,
                    "div_thr": float(self.div_thr),
                    "ratio_to_thr": None,
                    "loss_ema99": float(self.loss_ema99),
                    "ratio_to_ema": None,
                    "stat_condition": None,
                    "period_condition": bool(idx % self.skip_period == 0),
                    "ema_condition": None,
                    "reason": "inference_skip",
                }
            )
            return outputs

        outputs, features, box_features, class_logits = self._forward_detector(
            batched_inputs, need_loss=True
        )
        losses = self._adapt_losses(features, box_features, class_logits)
        total_loss = sum(losses[k] for k in losses) if losses else 0

        if self.skip_redundant is None:
            self.cur_used = True
        else:
            self.cur_used = False
            ga = float(losses["global_align"]) if "global_align" in losses else 0.0
            if "stat" in self.skip_redundant and ga > self.div_thr:
                self.cur_used = True
            elif "period" in self.skip_redundant and idx % self.skip_period == 0:
                self.cur_used = True
            elif "ema" in self.skip_redundant and ga / (self.loss_ema99 + 1e-7) > self.skip_beta:
                self.cur_used = True

        if torch.is_tensor(total_loss) and total_loss > 0 and self.cur_used:
            total_loss.backward()
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                torch.nn.utils.clip_grad_norm_(
                    self.model.backbone.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE,
                )
            self.optimizer.step()
            self.n_updates += 1
            updated = True
        else:
            self.pause_iter = 1
            updated = False

        if self.skip_redundant is not None:
            ga_value = float(losses["global_align"]) if "global_align" in losses else 0.0
            ratio_to_ema = ga_value / (self.loss_ema99 + 1e-7)
            stat_condition = (
                "stat" in self.skip_redundant and ga_value > self.div_thr
            )
            period_condition = (
                "period" in self.skip_redundant and idx % self.skip_period == 0
            )
            ema_condition = (
                "ema" in self.skip_redundant and ratio_to_ema > self.skip_beta
            )
            reason = "none"
            if stat_condition:
                reason = "stat"
            elif period_condition:
                reason = "period"
            elif ema_condition:
                reason = "ema"
            elif self.cur_used:
                reason = "forced_update"
            rec = {
                "iter": self.n_steps,
                "idx": idx,
                "statistics_evaluated": True,
                "updated": bool(updated),
                "pause_iter": self.pause_iter,
                "ga": ga_value,
                "div_thr": float(self.div_thr),
                "ratio_to_thr": ga_value / (self.div_thr + 1e-12),
                "loss_ema99": float(self.loss_ema99),
                "ratio_to_ema": ratio_to_ema,
                "stat_condition": bool(stat_condition),
                "period_condition": bool(period_condition),
                "ema_condition": bool(ema_condition),
                "reason": reason,
            }
            self._record_skip_diag(rec)

        self.optimizer.zero_grad(set_to_none=True)

        if "global_align" in losses:
            self.loss_ema99 = 0.99 * self.loss_ema99 + 0.01 * float(losses["global_align"])

        if self.n_steps <= self.diag_frames:
            self._prepost_diagnostic(batched_inputs, outputs, updated, losses)

        del losses, total_loss, features, box_features, class_logits
        return outputs

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _prepost_diagnostic(self, batched_inputs, pre_outputs, updated, losses):
        """
        Diagnostic only: re-run inference AFTER the update and measure how much
        the reported prediction would change.  Does not affect the evaluation,
        which always uses the native CTAOD pre-update output.
        """
        post_outputs, _, _, _ = self._forward_detector(batched_inputs, need_loss=False)
        pre = pre_outputs[0]["instances"]
        post = post_outputs[0]["instances"]

        def topk(inst, k=10):
            s = inst.scores
            n = min(k, len(s))
            return s[:n].cpu().numpy().tolist()

        rec = {
            "iter": self.n_steps,
            "updated": bool(updated),
            "loss": {k: float(v) for k, v in losses.items()},
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
            "n_pre_05": int((pre.scores > 0.5).sum()),
            "n_post_05": int((post.scores > 0.5).sum()),
            "top10_pre": topk(pre),
            "top10_post": topk(post),
        }
        if len(pre) > 0 and len(post) > 0:
            from detectron2.structures import pairwise_iou

            iou = pairwise_iou(pre.pred_boxes, post.pred_boxes)
            best_iou, best_j = iou.max(dim=1)
            same_cls = pre.pred_classes == post.pred_classes[best_j]
            rec["mean_best_iou"] = float(best_iou.mean())
            rec["frac_matched_iou09"] = float(((best_iou > 0.9) & same_cls).float().mean())
            rec["mean_abs_score_diff_matched"] = float(
                (pre.scores - post.scores[best_j]).abs().mean()
            )
            rec["max_abs_score_diff_matched"] = float(
                (pre.scores - post.scores[best_j]).abs().max()
            )
            rec["mean_abs_box_diff_matched"] = float(
                (pre.pred_boxes.tensor - post.pred_boxes.tensor[best_j]).abs().mean()
            )
        self.diag_records.append(rec)
        if self.n_steps == self.diag_frames:
            os.makedirs(os.path.dirname(self.diag_path), exist_ok=True)
            with open(self.diag_path, "w") as f:
                for r in self.diag_records:
                    f.write(json.dumps(r) + "\n")
            keys = [
                "mean_best_iou",
                "frac_matched_iou09",
                "mean_abs_score_diff_matched",
                "max_abs_score_diff_matched",
                "mean_abs_box_diff_matched",
            ]
            summary = {
                k: float(np.mean([r[k] for r in self.diag_records if k in r])) for k in keys
            }
            summary["frames"] = len(self.diag_records)
            summary["updated_frames"] = int(sum(r["updated"] for r in self.diag_records))
            summary["mean_n_pre_05"] = float(
                np.mean([r["n_pre_05"] for r in self.diag_records])
            )
            summary["mean_n_post_05"] = float(
                np.mean([r["n_post_05"] for r in self.diag_records])
            )
            msg = "CTAOD pre/post-update diagnostic: {}".format(json.dumps(summary))
            logger.info(msg)
            print(msg, flush=True)
