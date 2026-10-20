"""TENT for SemanticSegmentor (Wang et al. ICLR 2021).
Entropy minimization on BN affine params only; no EMA, no restore.
"""
from __future__ import annotations
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

__all__ = ["TENT_SemSeg"]


@META_ARCH_REGISTRY.register()
class TENT_SemSeg(nn.Module):
    """TENT wrapper around SemanticSegmentor: minimise softmax entropy online."""

    @configurable
    def __init__(self, *, model: nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.iter = 0

    @classmethod
    def from_config(cls, cfg):
        m = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
        DetectionCheckpointer(m).load(cfg.MODEL.WEIGHTS)
        m.to(torch.device(cfg.MODEL.DEVICE))
        m.train()
        # TENT: only BN affine params are trainable.
        m.requires_grad_(False)
        # Semantic FPN uses GroupNorm, not BatchNorm — update GN affine params.
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                mod.requires_grad_(True)
                if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    mod.track_running_stats = False
                    mod.running_mean = None
                    mod.running_var = None
        # Build optimizer from the now-filtered param set.
        optimizer = build_optimizer(cfg, m)
        return {"model": m, "optimizer": optimizer}

    @property
    def device(self):
        return self.model.pixel_mean.device

    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1
        images_norm = [
            ((x["image"].to(self.device).float() - self.model.pixel_mean) / self.model.pixel_std)
            for x in batched_inputs
        ]
        images = ImageList.from_tensors(
            images_norm,
            self.model.backbone.size_divisibility,
            padding_constraints=self.model.backbone.padding_constraints,
        )
        features = self.model.backbone(images.tensor)
        # Head must be eval so it returns logits not losses when targets=None.
        self.model.sem_seg_head.eval()
        logits, _ = self.model.sem_seg_head(features, None)
        probs = logits.float().softmax(dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        entropy.backward()
        self.optimizer.step()

        if self.iter % 50 == 0:
            print(f"[TENT_SemSeg] iter={self.iter} entropy={float(entropy.detach()):.4f}")

        with torch.no_grad():
            self.model.eval()
            processed_results = self.model(batched_inputs)
            self.model.train()
        return processed_results
