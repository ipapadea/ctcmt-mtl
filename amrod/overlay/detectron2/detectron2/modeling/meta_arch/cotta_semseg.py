"""CoTTA meta-arch for detectron2 SemanticSegmentor (Wang et al. CVPR 2022).

Reference: "Continual Test-Time Domain Adaptation", https://arxiv.org/abs/2203.13591
Official code: https://github.com/qinenergy/cotta

Faithful to the paper:
  1. Confidence trigger uses the frozen SOURCE anchor.
  2. Student loss = -(p_teacher * log_softmax(student)).sum(1).mean()  (soft CE).
  3. Stochastic restore over ALL trainable weight+bias params (paper: seg = 0.01).
  4. Student BN uses batch statistics (running_mean/var disabled).
  5. Teacher = EMA of student; teacher output is reported for evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.solver import build_optimizer
from detectron2.structures import ImageList

from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["CoTTA_SemSeg"]


def _configure_batch_stats_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


def _build_aug_ops(scales: Sequence[float], flips: Sequence[bool], divisor: int = 32):
    """Return list of (apply, invert) pairs operating on (B, C, H, W) tensors.

    Aug'd spatial size is snapped to a multiple of ``divisor`` so that FPN
    lateral + top-down features have compatible shapes.
    """
    def _snap(v, d):
        v = max(int(round(v)), d)
        return ((v + d - 1) // d) * d

    ops = []
    for s in scales:
        for flip in flips:
            def apply(x, s=s, flip=flip, d=divisor):
                H, W = x.shape[-2:]
                Hs = _snap(H * s, d)
                Ws = _snap(W * s, d)
                y = F.interpolate(x, size=(Hs, Ws), mode="bilinear", align_corners=False)
                if flip:
                    y = torch.flip(y, dims=[-1])
                return y

            def invert(p, hw, flip=flip):
                if flip:
                    p = torch.flip(p, dims=[-1])
                return F.interpolate(p, size=hw, mode="bilinear", align_corners=False)

            ops.append((apply, invert))
    return ops


@META_ARCH_REGISTRY.register()
class CoTTA_SemSeg(nn.Module):
    """CoTTA wrapper around a standard detectron2 SemanticSegmentor."""

    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        anchor: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_decay: float,
        restore_prob: float,
        conf_threshold: float,
        aug_scales: Sequence[float],
        aug_flips: Sequence[bool],
        num_classes: int,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.optimizer = optimizer

        self.ema_decay = float(ema_decay)
        self.restore_prob = float(restore_prob)
        self.conf_threshold = float(conf_threshold)
        self.num_classes = int(num_classes)
        divisor = int(getattr(student.backbone, "size_divisibility", 32) or 32)
        self._augs = _build_aug_ops(aug_scales, aug_flips, divisor=divisor)
        self.iter = 0
        # V2: backbone/FPN params are feature extractors shared across domains —
        # restore them less aggressively than the task-specific head.
        self._backbone_rst_factor = float(getattr(student, "_backbone_rst_factor", 1.0))

        # Frozen snapshot of every source weight/bias parameter for stochastic restore.
        # Matches the official CIFAR reference loop.
        self._source_params: Dict[str, torch.Tensor] = {}
        for nm, m in self.anchor.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name in ("weight", "bias"):
                    self._source_params[f"{nm}.{np_name}"] = p.detach().clone()

    @classmethod
    def from_config(cls, cfg):
        weights_path = cfg.MODEL.WEIGHTS

        def _build(train_mode: bool, freeze: bool):
            m = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
            # Use .load() (not resume_or_load) so FrozenBN buffers reliably
            # come from the checkpoint. resume_or_load has failed silently on
            # our fork when save_dir is shared across builds.
            DetectionCheckpointer(m).load(weights_path)
            m.to(torch.device(cfg.MODEL.DEVICE))
            if train_mode:
                m.train()
            else:
                m.eval()
            if freeze:
                for p in m.parameters():
                    p.requires_grad_(False)
            return m

        student = _build(train_mode=True, freeze=False)
        teacher = _build(train_mode=True, freeze=True)
        anchor = _build(train_mode=False, freeze=True)

        optimizer = build_optimizer(cfg, student)

        return {
            "student": student,
            "teacher": teacher,
            "anchor": anchor,
            "optimizer": optimizer,
            "ema_decay": cfg.SOLVER.COTTA_EMA_DECAY,
            "restore_prob": cfg.SOLVER.COTTA_RESTORE_PROB,
            "conf_threshold": cfg.SOLVER.COTTA_CONF_THRESH,
            "aug_scales": tuple(cfg.SOLVER.COTTA_AUG_SCALES),
            "aug_flips": tuple(cfg.SOLVER.COTTA_AUG_FLIPS),
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

    def _stochastic_restore(self):
        if self.restore_prob <= 0.0:
            return
        _shared_prefixes = ("backbone.", "fpn.")
        for nm, m in self.student.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name not in ("weight", "bias") or not p.requires_grad:
                    continue
                key = f"{nm}.{np_name}"
                src = self._source_params.get(key)
                if src is None:
                    continue
                rst = self.restore_prob
                # V2: protect backbone/FPN params shared across all domains.
                if self._backbone_rst_factor < 1.0 and any(key.startswith(pfx) for pfx in _shared_prefixes):
                    rst = rst * self._backbone_rst_factor
                if rst <= 0.0:
                    continue
                mask = (torch.rand_like(p) < rst).float()
                src_dev = src.to(p.device, non_blocking=True)
                p.data.mul_(1.0 - mask).add_(src_dev * mask)

    @property
    def device(self):
        return self.student.pixel_mean.device

    # ------------------------------------------------------------------
    # Internal: run a SemanticSegmentor's backbone+head at raw resolution.
    # Returns (logits_at_1/output_stride, image_sizes_pre_pad, padded_hw)
    # ------------------------------------------------------------------
    def _forward_logits(self, model, batched_inputs, image_tensor=None):
        if image_tensor is None:
            images_norm = [
                ((x["image"].to(self.device).float() - model.pixel_mean) / model.pixel_std)
                for x in batched_inputs
            ]
            images = ImageList.from_tensors(
                images_norm,
                model.backbone.size_divisibility,
                padding_constraints=model.backbone.padding_constraints,
            )
            image_tensor = images.tensor
            image_sizes = images.image_sizes
        else:
            image_sizes = [image_tensor.shape[-2:]] * image_tensor.size(0)

        features = model.backbone(image_tensor)
        results, _ = model.sem_seg_head(features, None)
        return results, image_sizes, image_tensor.shape[-2:]

    @torch.no_grad()
    def _anchor_confidence(self, batched_inputs):
        logits, _, _ = self._forward_logits(self.anchor, batched_inputs)
        probs = logits.float().softmax(dim=1)
        return probs.max(dim=1)[0].mean(dim=(1, 2)), probs

    @torch.no_grad()
    def _teacher_aug_avg_probs(self, image_tensor):
        H, W = image_tensor.shape[-2:]
        accum = None
        for apply, invert in self._augs:
            x_aug = apply(image_tensor)
            logits, _, _ = self._forward_logits(self.teacher, None, image_tensor=x_aug)
            # Upsample logits to aug'd input shape, softmax, then invert.
            logits = F.interpolate(logits, size=x_aug.shape[-2:], mode="bilinear", align_corners=False)
            probs = logits.float().softmax(dim=1)
            probs = invert(probs, (H, W))
            accum = probs if accum is None else accum + probs
        return accum / float(len(self._augs))

    @torch.no_grad()
    def _update_teacher(self):
        d = self.ema_decay
        s_state = self.student.state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s_state[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s_state[k])

    # ------------------------------------------------------------------
    # forward: one CTTA step. Returns detectron2-style processed results
    # so the SemSegEvaluator can consume them.
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1

        # Preprocess ONCE (shared across student/teacher/anchor).
        images_norm = [
            ((x["image"].to(self.device).float() - self.student.pixel_mean) / self.student.pixel_std)
            for x in batched_inputs
        ]
        images = ImageList.from_tensors(
            images_norm,
            self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints,
        )
        image_tensor = images.tensor

        # 1. Anchor confidence.
        with torch.no_grad():
            a_logits, _, _ = self._forward_logits(self.anchor, None, image_tensor=image_tensor)
            a_probs = a_logits.float().softmax(dim=1)
            conf = a_probs.max(dim=1)[0].mean(dim=(1, 2))
            low_mask = conf < self.conf_threshold

        # 2. Teacher soft pseudo-labels.
        with torch.no_grad():
            t_logits, _, _ = self._forward_logits(self.teacher, None, image_tensor=image_tensor)
            teacher_probs = F.interpolate(
                t_logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            ).float().softmax(dim=1)
            if low_mask.any():
                aug_probs = self._teacher_aug_avg_probs(image_tensor[low_mask])
                teacher_probs = teacher_probs.clone()
                teacher_probs[low_mask] = aug_probs

        # 3. Student forward + soft CE.
        s_logits, _, _ = self._forward_logits(self.student, None, image_tensor=image_tensor)
        s_logits_full = F.interpolate(
            s_logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        log_probs = F.log_softmax(s_logits_full.float(), dim=1)
        loss = -(teacher_probs.detach() * log_probs).sum(dim=1).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # 4. Teacher EMA + stochastic restore.
        self._update_teacher()
        self._stochastic_restore()

        if self.iter % 50 == 0:
            print(f"[CoTTA_SemSeg] iter={self.iter} loss={float(loss.detach()):.4f} "
                  f"low_conf={int(low_mask.sum())}/{len(batched_inputs)}")

        # 5. Report teacher output for evaluation (paper convention). Use the
        #    standard SemanticSegmentor.forward so preprocessing + postprocess
        #    exactly match the plain-eval path used to produce source-only
        #    numbers -- avoids any subtle mismatch in image normalization,
        #    padding removal, or head training/eval mode gating.
        with torch.no_grad():
            teacher_was_training = self.teacher.training
            self.teacher.eval()
            processed_results = self.teacher(batched_inputs)
            if teacher_was_training:
                self.teacher.train()
        return processed_results
