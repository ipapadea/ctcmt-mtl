"""CTCMT_Seg: our method applied to single-task semantic segmentation.
Same adaptation principles as CTCMT_MTL's seg branch, but wraps
SemanticSegmentor so it can be used with a dedicated Semantic FPN source.

Components (our contributions, NOT CoTTA):
  1. Soft-CE consistency from EMA teacher (same as CTCMT_MTL seg branch)
  2. V2: backbone-protected stochastic restore
  3. V4: per-class seg feature prototype anchor (anti-forgetting)

Contrast with CoTTA (what we do NOT do):
  - No augmentation-averaged pseudo-labels (14 scales)
  - No confidence-threshold gating
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.solver import build_optimizer
from detectron2.structures import ImageList

from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["CTCMT_Seg"]


@META_ARCH_REGISTRY.register()
class CTCMT_Seg(nn.Module):
    """Our seg-only CTTA adapter wrapping SemanticSegmentor."""

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
        backbone_rst_factor: float,
        num_classes: int,
        proto_ema: float,
        proto_weight: float,
        proto_conf_thresh: float,
        aug_enabled: bool = False,
        aug_scales: tuple = (1.0,),
        aug_flips: tuple = (False,),
        aug_conf_thresh: float = 0.9,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.optimizer = optimizer
        self.ema_decay = float(ema_decay)
        self.restore_prob = float(restore_prob)
        self.backbone_rst_factor = float(backbone_rst_factor)
        self.num_classes = int(num_classes)
        self.proto_ema = float(proto_ema)
        self.proto_weight = float(proto_weight)
        self.proto_conf_thresh = float(proto_conf_thresh)
        divisor = int(getattr(student.backbone, "size_divisibility", 32) or 32)
        self._aug_enabled = bool(aug_enabled)
        self._aug_scales = tuple(aug_scales)
        self._aug_flips = tuple(aug_flips)
        self._aug_conf_thresh = float(aug_conf_thresh)
        self._divisor = divisor
        # source_proto_init: build prototypes from anchor on first image and freeze.
        self._source_proto_init = bool(getattr(student, "_source_proto_init", False))
        self._source_proto_frozen = False
        self.iter = 0

        # Source snapshot for stochastic restore.
        self._source_params: Dict[str, torch.Tensor] = {}
        for nm, m in self.anchor.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name in ("weight", "bias"):
                    self._source_params[f"{nm}.{np_name}"] = p.detach().clone()

        # Per-class feature prototypes (seg-only, updated by teacher predictions).
        self._prototypes: Dict[int, torch.Tensor] = {}

    @classmethod
    def from_config(cls, cfg):
        weights = cfg.MODEL.WEIGHTS

        def _build(train_mode, freeze):
            m = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
            DetectionCheckpointer(m).load(weights)
            m.to(torch.device(cfg.MODEL.DEVICE))
            if train_mode:
                m.train()
            else:
                m.eval()
            if freeze:
                for p in m.parameters():
                    p.requires_grad_(False)
            return m

        student = _build(True, False)
        teacher = _build(True, True)
        anchor = _build(False, True)
        optimizer = build_optimizer(cfg, student)
        obj = {
            "student": student,
            "teacher": teacher,
            "anchor": anchor,
            "optimizer": optimizer,
            "ema_decay": cfg.SOLVER.COTTA_EMA_DECAY,
            "restore_prob": cfg.SOLVER.COTTA_RESTORE_PROB,
            "backbone_rst_factor": float(getattr(cfg.SOLVER, "CTCMT_BACKBONE_RST_FACTOR", 1.0)),
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "proto_ema": float(getattr(cfg.SOLVER, "CTCMT_PROTO_EMA", 0.999)),
            "proto_weight": float(getattr(cfg.SOLVER, "CTCMT_PROTO_WEIGHT", 0.0)),
            "proto_conf_thresh": float(getattr(cfg.SOLVER, "CTCMT_SEG_PROTO_CONF_THRESH", 0.9)),
            "aug_enabled": bool(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_ENABLED", False)),
            "aug_scales": tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_SCALES", (1.0,))),
            "aug_flips": tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_FLIPS", (False,))),
            "aug_conf_thresh": float(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_CONF_THRESH", 0.9)),
        }
        # source_proto_init flag stored on student temporarily so __init__ can read it.
        student._source_proto_init = bool(getattr(cfg.SOLVER, "CTCMT_SEG_SOURCE_PROTO", False))
        return obj

    def _maybe_init_source_protos(self, image_tensor, feat_key, feat):
        """On first call, build prototypes from the frozen anchor model and freeze them."""
        if not self._source_proto_init or self._source_proto_frozen:
            return
        with torch.no_grad():
            a_logits, a_feats = self._forward_logits(self.anchor, image_tensor)
            a_probs = F.interpolate(a_logits, size=image_tensor.shape[-2:],
                                    mode="bilinear", align_corners=False).float().softmax(dim=1)
            a_feat = a_feats[feat_key]
            H, W = a_feat.shape[-2:]
            a_probs_ds = F.interpolate(a_probs, size=(H, W),
                                        mode="bilinear", align_corners=False)
            for c in range(self.num_classes):
                mask = (a_probs_ds[0, c] >= self.proto_conf_thresh)
                if mask.sum() < 4:
                    continue
                z = a_feat[0, :, mask].mean(dim=1)
                self._prototypes[c] = F.normalize(z, dim=0).detach().clone()
            self._source_proto_frozen = True

    @property
    def device(self):
        return self.student.pixel_mean.device

    def _forward_logits(self, model, image_tensor):
        features = model.backbone(image_tensor)
        model.sem_seg_head.eval()
        logits, _ = model.sem_seg_head(features, None)
        return logits, features

    @torch.no_grad()
    def _aug_avg_teacher_probs(self, image_tensor):
        """Augmentation-averaged teacher predictions (same math as CoTTA)."""
        H, W = image_tensor.shape[-2:]
        d = self._divisor
        def snap(v):
            v = max(int(round(v)), d)
            return ((v + d - 1) // d) * d
        accum = None
        for s in self._aug_scales:
            Hs, Ws = snap(H * s), snap(W * s)
            for flip in self._aug_flips:
                x = F.interpolate(image_tensor, size=(Hs, Ws),
                                  mode="bilinear", align_corners=False)
                if flip:
                    x = torch.flip(x, dims=[-1])
                feats = self.teacher.backbone(x)
                self.teacher.sem_seg_head.eval()
                logits, _ = self.teacher.sem_seg_head(feats, None)
                logits = F.interpolate(logits, size=(Hs, Ws),
                                       mode="bilinear", align_corners=False)
                probs = logits.float().softmax(dim=1)
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                probs = F.interpolate(probs, size=(H, W),
                                      mode="bilinear", align_corners=False)
                accum = probs if accum is None else accum + probs
        return accum / float(max(len(self._aug_scales) * len(self._aug_flips), 1))

    @torch.no_grad()
    def _update_teacher(self):
        d = self.ema_decay
        s = self.student.state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s[k])

    @torch.no_grad()
    def _stochastic_restore(self):
        _shared = ("backbone.", "fpn.")
        for nm, m in self.student.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name not in ("weight", "bias") or not p.requires_grad:
                    continue
                key = f"{nm}.{np_name}"
                src = self._source_params.get(key)
                if src is None:
                    continue
                rst = self.restore_prob
                if self.backbone_rst_factor < 1.0 and any(key.startswith(pfx) for pfx in _shared):
                    rst *= self.backbone_rst_factor
                if rst <= 0.0:
                    continue
                mask = (torch.rand_like(p) < rst).float()
                p.data.mul_(1.0 - mask).add_(src.to(p.device) * mask)

    def _proto_loss(self, student_features, teacher_probs, image_tensor=None):
        """Pull current backbone features toward per-class EMA prototypes."""
        if self.proto_weight <= 0.0:
            return student_features[list(student_features.keys())[-1]].new_zeros(())
        feat_key = list(student_features.keys())[-1]
        feat = student_features[feat_key]  # (1, C, H, W)
        # Initialize source prototypes from anchor on the very first call.
        if self._source_proto_init and not self._source_proto_frozen and image_tensor is not None:
            self._maybe_init_source_protos(image_tensor, feat_key, feat)
        C = feat.size(1)
        probs_ds = F.interpolate(teacher_probs, size=feat.shape[-2:],
                                 mode="bilinear", align_corners=False)
        loss = feat.new_zeros(())
        for c in range(self.num_classes):
            conf_mask = (probs_ds[0, c] >= self.proto_conf_thresh)  # (H, W)
            if conf_mask.sum() < 4:
                continue
            # Class-c feature centroid from current student features.
            z = feat[0, :, conf_mask].mean(dim=1)   # (C,)
            z_n = F.normalize(z, dim=0)
            with torch.no_grad():
                if c not in self._prototypes:
                    self._prototypes[c] = z_n.detach().clone()
                else:
                    a = self.proto_ema
                    self._prototypes[c] = a * self._prototypes[c] + (1 - a) * z_n.detach()
            proto = self._prototypes[c].to(feat.device)
            loss = loss + (1.0 - (z_n * proto.detach()).sum())
        return loss

    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1
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

        # Teacher soft pseudo-labels.
        with torch.no_grad():
            t_logits, _ = self._forward_logits(self.teacher, image_tensor)
            teacher_probs = F.interpolate(
                t_logits, size=image_tensor.shape[-2:],
                mode="bilinear", align_corners=False,
            ).float().softmax(dim=1)
            # Augmentation-averaged teacher when anchor is uncertain.
            if self._aug_enabled:
                a_logits, _ = self._forward_logits(self.anchor, image_tensor)
                a_conf = a_logits.float().softmax(dim=1).max(dim=1)[0].mean()
                if float(a_conf.item()) < self._aug_conf_thresh:
                    teacher_probs = self._aug_avg_teacher_probs(image_tensor)

        # Student forward.
        s_feats = self.student.backbone(image_tensor)
        self.student.sem_seg_head.eval()
        s_logits, _ = self.student.sem_seg_head(s_feats, None)
        s_logits_full = F.interpolate(s_logits, size=image_tensor.shape[-2:],
                                       mode="bilinear", align_corners=False)

        # Soft-CE consistency (our core seg loss).
        s_log_probs = F.log_softmax(s_logits_full.float(), dim=1)
        loss = -(teacher_probs.detach() * s_log_probs).sum(dim=1).mean()

        # V4 prototype anchor (anti-forgetting).
        if self.proto_weight > 0.0:
            proto_l = self._proto_loss(s_feats, teacher_probs.detach(), image_tensor)
            loss = loss + self.proto_weight * proto_l

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._update_teacher()
        self._stochastic_restore()

        if self.iter % 50 == 0:
            print(f"[CTCMT_Seg] iter={self.iter} loss={float(loss.detach()):.4f}")

        with torch.no_grad():
            self.teacher.eval()
            results = self.teacher(batched_inputs)
            self.teacher.train()
        return results
