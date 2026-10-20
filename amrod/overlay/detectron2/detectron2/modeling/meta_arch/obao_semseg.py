"""OBAO semantic segmentation (ECCV 2024), same-source Panoptic-FPN port."""
from __future__ import annotations

import logging
import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils.run_provenance import log_run_provenance

from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["OBAO_SemSeg"]

logger = logging.getLogger(__name__)


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def _build_aug_ops(scales: Sequence[float], flips: Sequence[bool], divisor: int):
    def _snap(value, multiple):
        value = max(int(round(value)), multiple)
        return ((value + multiple - 1) // multiple) * multiple

    ops = []
    for scale in scales:
        for flip in flips:
            def apply(x, scale=scale, flip=flip):
                height, width = x.shape[-2:]
                size = (_snap(height * scale, divisor), _snap(width * scale, divisor))
                y = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
                return torch.flip(y, dims=[-1]) if flip else y

            def invert(probs, size, flip=flip):
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                return F.interpolate(probs, size=size, mode="bilinear", align_corners=False)

            ops.append((float(scale), bool(flip), apply, invert))
    return ops


def _topology_preserving_loss(source_relation, current_relation):
    numerator = (source_relation * current_relation).sum()
    denominator = torch.sqrt(
        source_relation.square().sum() * current_relation.square().sum()
    )
    return -numerator / denominator


@META_ARCH_REGISTRY.register()
class OBAO_SemSeg(nn.Module):
    """Official OBAO segmentation loop around a detectron2 SemanticSegmentor."""

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
        anchor_confidence: float,
        entropy_margin: float,
        pce_weight: float,
        topology_weight: float,
        aug_scales: Sequence[float],
        aug_flips: Sequence[bool],
        cfg,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.optimizer = optimizer
        self.ema_decay = float(ema_decay)
        self.restore_prob = float(restore_prob)
        self.anchor_confidence = float(anchor_confidence)
        self.entropy_margin = float(entropy_margin)
        self.pce_weight = float(pce_weight)
        self.topology_weight = float(topology_weight)
        self.num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.iter = 0

        head = self.student.sem_seg_head
        if not all(hasattr(head, name) for name in ("in_features", "scale_heads", "predictor")):
            raise TypeError("OBAO_SemSeg requires a SemSegFPNHead-compatible semantic head")

        divisor = int(getattr(self.student.backbone, "size_divisibility", 32) or 32)
        self._augs = _build_aug_ops(aug_scales, aug_flips, divisor)
        if not any(scale == 1.0 and not flip for scale, flip, _, _ in self._augs):
            raise ValueError("OBAO augmentation set must contain the unflipped 1.0-scale view")

        source_weights = self.anchor.sem_seg_head.predictor.weight.detach().flatten(1)
        source_weights = F.normalize(source_weights, dim=1)
        self.register_buffer("source_relation", source_weights @ source_weights.T, persistent=False)

        self._source_params: Dict[str, torch.Tensor] = {}
        for module_name, module in self.anchor.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name in ("weight", "bias"):
                    key = f"{module_name}.{parameter_name}"
                    self._source_params[key] = parameter.detach().clone()

        trainable = [(name, p) for name, p in self.student.named_parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for _, p in trainable)
        frozen_student = [name for name, p in self.student.named_parameters() if not p.requires_grad]
        if frozen_student:
            raise AssertionError(f"all registered student parameters must be trainable: {frozen_student}")
        if any(p.requires_grad for p in self.teacher.parameters()):
            raise AssertionError("OBAO EMA teacher must be frozen")
        if any(p.requires_grad for p in self.anchor.parameters()):
            raise AssertionError("OBAO source anchor must be frozen")

        log_run_provenance(
            cfg,
            method="OBAO-seg",
            extra={
                "semantic_branch_only": True,
                "detection_branch_used": False,
                "trainable_parameter_tensors": len(trainable),
                "trainable_parameters": trainable_count,
                "optimizer": "Adam",
                "optimizer_lr": optimizer.param_groups[0]["lr"],
                "ema_decay": self.ema_decay,
                "restore_prob": self.restore_prob,
                "anchor_confidence": self.anchor_confidence,
                "entropy_margin_factor": self.entropy_margin / math.log(self.num_classes),
                "entropy_margin": self.entropy_margin,
                "pce_weight": self.pce_weight,
                "topology_weight": self.topology_weight,
                "augmentation_views": len(self._augs),
            },
            logger=logger,
        )

    @classmethod
    def from_config(cls, cfg):
        def build_source(frozen):
            model = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            model.to(torch.device(cfg.MODEL.DEVICE))
            model.eval()
            model.requires_grad_(not frozen)
            return model

        student = build_source(frozen=False)
        teacher = build_source(frozen=True)
        anchor = build_source(frozen=True)
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=cfg.SOLVER.OBAO_LR,
            betas=(0.9, 0.999),
        )
        return {
            "student": student,
            "teacher": teacher,
            "anchor": anchor,
            "optimizer": optimizer,
            "ema_decay": cfg.SOLVER.OBAO_EMA_DECAY,
            "restore_prob": cfg.SOLVER.OBAO_RESTORE_PROB,
            "anchor_confidence": cfg.SOLVER.OBAO_ANCHOR_CONFIDENCE,
            "entropy_margin": math.log(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
            * cfg.SOLVER.OBAO_MARGIN,
            "pce_weight": cfg.SOLVER.OBAO_PCE_WEIGHT,
            "topology_weight": cfg.SOLVER.OBAO_TOPOLOGY_WEIGHT,
            "aug_scales": tuple(cfg.SOLVER.OBAO_AUG_SCALES),
            "aug_flips": tuple(cfg.SOLVER.OBAO_AUG_FLIPS),
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.student.pixel_mean.device

    def _preprocess(self, batched_inputs):
        normalized = [
            (sample["image"].to(self.device).float() - self.student.pixel_mean)
            / self.student.pixel_std
            for sample in batched_inputs
        ]
        return ImageList.from_tensors(
            normalized,
            self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints,
        )

    @staticmethod
    def _forward_raw(model, image_tensor):
        features = model.backbone(image_tensor)
        head = model.sem_seg_head
        fused = None
        for index, feature_name in enumerate(head.in_features):
            current = head.scale_heads[index](features[feature_name])
            fused = current if index == 0 else fused + current
        return head.predictor(fused), fused

    @torch.no_grad()
    def _augmented_teacher_probs(self, image_tensor):
        original_size = image_tensor.shape[-2:]
        average = None
        default = None
        for scale, flip, apply, invert in self._augs:
            augmented = apply(image_tensor)
            logits, _ = self._forward_raw(self.teacher, augmented)
            logits = F.interpolate(
                logits, size=augmented.shape[-2:], mode="bilinear", align_corners=False
            )
            probs = invert(logits.float().softmax(dim=1), original_size)
            average = probs if average is None else average + probs
            if scale == 1.0 and not flip:
                default = probs
        return average / len(self._augs), default

    @torch.no_grad()
    def _anchor_default_probs(self, image_tensor):
        logits, _ = self._forward_raw(self.anchor, image_tensor)
        logits = F.interpolate(
            logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits.float().softmax(dim=1)

    @torch.no_grad()
    def _update_teacher(self):
        for teacher_parameter, student_parameter in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_parameter.mul_(self.ema_decay).add_(
                student_parameter.detach(), alpha=1.0 - self.ema_decay
            )

    @torch.no_grad()
    def _stochastic_restore(self):
        if self.restore_prob <= 0:
            return
        for module_name, module in self.student.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name not in ("weight", "bias") or not parameter.requires_grad:
                    continue
                key = f"{module_name}.{parameter_name}"
                source = self._source_params.get(key)
                if source is None:
                    continue
                mask = torch.rand_like(parameter) < self.restore_prob
                parameter.copy_(torch.where(mask, source.to(parameter.device), parameter))

    def _auxiliary_losses(self, student_logits, student_features):
        entropy = _softmax_entropy(student_logits)
        low_entropy = entropy < self.entropy_margin
        if not low_entropy.any():
            zero = student_logits.sum() * 0.0
            return zero, zero, 0, 0

        flat_logits = student_logits.permute(0, 2, 3, 1)[low_entropy]
        flat_features = student_features.permute(0, 2, 3, 1)[low_entropy]
        pseudo_labels = flat_logits.argmax(dim=1)
        pce_loss = F.cross_entropy(flat_logits, pseudo_labels)

        prototypes = []
        present_classes = []
        for class_id in range(self.num_classes):
            selected = pseudo_labels == class_id
            if selected.any():
                prototypes.append(flat_features[selected].mean(dim=0))
                present_classes.append(class_id)

        prototypes = F.normalize(torch.stack(prototypes), dim=1)
        current_relation = prototypes @ prototypes.T
        class_index = torch.tensor(present_classes, device=self.device, dtype=torch.long)
        source_relation = self.source_relation[class_index][:, class_index]
        topology_loss = _topology_preserving_loss(source_relation, current_relation)
        return pce_loss, topology_loss, int(low_entropy.sum()), len(present_classes)

    def _format_preupdate_results(self, pseudo_labels, batched_inputs, image_sizes):
        scores = F.one_hot(pseudo_labels, num_classes=self.num_classes).permute(0, 3, 1, 2)
        results = []
        for score, sample, image_size in zip(scores.float(), batched_inputs, image_sizes):
            height = sample.get("height", image_size[0])
            width = sample.get("width", image_size[1])
            results.append(
                {"sem_seg": sem_seg_postprocess(score, image_size, height, width)}
            )
        return results

    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1
        images = self._preprocess(batched_inputs)
        image_tensor = images.tensor

        with torch.no_grad():
            augmented_probs, teacher_default = self._augmented_teacher_probs(image_tensor)
            anchor_default = self._anchor_default_probs(image_tensor)
            confident_source = (
                anchor_default.max(dim=1).values > self.anchor_confidence
            )
            teacher_default_labels = teacher_default.argmax(dim=1)
            augmented_labels = augmented_probs.argmax(dim=1)
            pseudo_labels = torch.where(
                confident_source, teacher_default_labels, augmented_labels
            )
            processed_results = self._format_preupdate_results(
                pseudo_labels, batched_inputs, images.image_sizes
            )

        student_logits, student_features = self._forward_raw(self.student, image_tensor)
        full_logits = F.interpolate(
            student_logits,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        targets = pseudo_labels.clone()
        for batch_index, (height, width) in enumerate(images.image_sizes):
            targets[batch_index, height:, :] = self.student.sem_seg_head.ignore_value
            targets[batch_index, :, width:] = self.student.sem_seg_head.ignore_value
        base_loss = F.cross_entropy(
            full_logits.float(),
            targets,
            ignore_index=self.student.sem_seg_head.ignore_value,
        )
        pce_loss, topology_loss, low_pixels, present_classes = self._auxiliary_losses(
            student_logits, student_features
        )
        total_loss = (
            base_loss
            + self.pce_weight * pce_loss
            + self.topology_weight * topology_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self._update_teacher()
        self._stochastic_restore()

        if self.iter <= 5 or self.iter % 50 == 0:
            logger.info(
                "OBAO-seg iter=%d loss=%.6f base=%.6f pce=%.6f topology=%.6f "
                "low_pixels=%d present_classes=%d source_confident=%.4f",
                self.iter,
                float(total_loss.detach()),
                float(base_loss.detach()),
                float(pce_loss.detach()),
                float(topology_loss.detach()),
                low_pixels,
                present_classes,
                float(confident_source.float().mean()),
            )

        return processed_results
