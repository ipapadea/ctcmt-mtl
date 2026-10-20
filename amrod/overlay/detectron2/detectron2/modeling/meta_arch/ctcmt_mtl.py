"""CT-CMT-MTL adapter for detectron2 PanopticFPN.

Extends the published CT-CMT idea (Moraiti et al. 2026, single-task detection
via YOLOX) to multi-task learning on Panoptic FPN (det + sem-seg).

Three loss terms during adaptation, all computed from the STUDENT model and
supervised by the frozen TEACHER (EMA of student):

  1. Detection consistency: teacher inference -> boxes + classes filtered by
     dynamic per-class thresholds (Wei et al. 2024 style, matching AMROD).
     Student is trained with these as gt_instances via the standard Faster
     R-CNN proposal + roi head losses. Mask head is disabled during adaptation
     (no reliable pseudo-masks).

  2. Semantic segmentation consistency: soft cross-entropy of student
     sem_seg logits against teacher sem_seg softmax (matches CoTTA).

  3. Cross-task supervised contrastive (CT-CL, paper's core contribution):
     For each pseudo-detected box b with class c, build TWO feature views
     from the student FPN feature maps:

         z_det(b) = RoI-Align at b                             # det view
         z_seg(b) = mask-weighted mean inside b, weighted by
                     teacher_seg_probs[..., c][b]              # seg view

     Same-class pairs (det<->det, seg<->seg, det<->seg) are pulled together;
     different-class pairs pushed apart. Creates a shared cross-task
     representation.

Teacher weights are updated as EMA of student. All trainable weight+bias
params in the student are stochastically restored to source with prob
``COTTA_RESTORE_PROB`` per step (CoTTA convention).

Reference:
    Moraiti et al. 2026 (EJAI) --- single-task CT-CMT for YOLOX.
    Wang et al. 2022 (CVPR) --- CoTTA (segmentation CTTA).
    Wei et al. 2024 --- AMROD (dynamic per-class thresholds).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Sequence

import json
import math
import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.solver import build_optimizer
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage

from ..postprocessing import detector_postprocess, sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["CTCMT_MTL"]

# Cityscapes taxonomy: detection classes 0..7 (thing_classes) map onto
# semantic-segmentation trainIds 11..18 (person, rider, car, truck, bus,
# train, motorcycle, bicycle). Used for the CT-CL seg view when computing
# mask-weighted features per bbox class.
_DET_TO_SEG_CLASS_CITYSCAPES = (11, 12, 13, 14, 15, 16, 17, 18)

# Parameters shared by both task branches in PanopticFPN: the ResNet bottom-up
# stack and the FPN necks live under ``backbone.``; ``proposal_generator.`` and
# ``roi_heads.`` are detection-only, ``sem_seg_head.`` is segmentation-only.
_SHARED_PARAM_PREFIXES = ("backbone.",)

# ResNet encoder only. Detectron2's Panoptic-FPN keeps both the bottom-up
# ResNet and the FPN under ``backbone.``, so this narrower prefix is needed
# for ResNet-block / FPN-open gradient routing.
_RESNET_PARAM_PREFIXES = ("backbone.bottom_up.",)

# _VALID_CONFLICT_MODES = ("none", "protect_det", "cagrad", "hard_decouple",
                        #  "dyn_weight", "aux_head_only")
_VALID_CONFLICT_MODES = (
    "none",
    "protect_det",
    "cagrad",
    "hard_decouple",
    "dyn_weight",
    "aux_head_only",
    "resnet_route",
    "task_coco",
)

# =====================================================================
# Supervised contrastive loss (Khosla et al. 2020) --- compact inline.
# =====================================================================
def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """features: (N, D) L2-normalized. labels: (N,) int."""
    N = features.size(0)
    if N < 2:
        return features.new_zeros(())
    device = features.device
    # Similarity matrix.
    logits = torch.matmul(features, features.t()) / temperature
    # For numerical stability.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    # Positive mask (same class), excluding self.
    labels = labels.view(-1, 1)
    mask_pos = torch.eq(labels, labels.t()).float().to(device)
    diag = torch.eye(N, device=device)
    mask_pos = mask_pos - diag
    exp_logits = torch.exp(logits) * (1.0 - diag)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    n_pos = mask_pos.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return features.new_zeros(())
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


# =====================================================================
# BN convention (TENT / CoTTA)
# =====================================================================
def _configure_batch_stats_bn(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None


# =====================================================================
# Dynamic threshold (Wei et al. 2024 style, same math as AMROD)
# =====================================================================
def _dyn_thresholds(prev, per_class_mean_scores, alpha: float, gamma: float,
                    lo: float, hi: float):
    new = []
    for th, mean in zip(prev, per_class_mean_scores):
        if mean > 0:
            th = gamma * th + (1.0 - gamma) * alpha * math.sqrt(mean)
        th = max(min(th, hi), lo)
        new.append(th)
    return new


# =====================================================================
# Gradient-magnitude restore mask (ported from amrod.py find_weight_quantile).
# =====================================================================
def _fisher_restore_mask(fisher: torch.Tensor, perc: float) -> torch.Tensor:
    """Binary mask selecting the ``perc`` least-important entries of ``fisher``.

    Importance is the squared gradient (empirical Fisher diagonal), scaled by
    uniform noise so the selection stays stochastic rather than deterministic.
    """
    if perc <= 0.0:
        return torch.zeros_like(fisher)
    if perc >= 1.0:
        return torch.ones_like(fisher)
    weights = fisher / fisher.max().clamp_min(1e-12)
    noisy = weights * torch.rand_like(weights)
    flat = torch.sort(noisy.reshape(-1)).values
    n = flat.numel()
    frac_idx = perc * (n - 1)
    low = int(frac_idx)
    high = min(low + 1, n - 1)
    thresh = flat[low] + (flat[high] - flat[low]) * (frac_idx - low)
    return (noisy < thresh).float()


# =====================================================================
# Gradient-conflict utilities (S1/S2/S3/S5 screening batch).
# =====================================================================
def _shared_block_of(name: str) -> str:
    """Coarse block label for a shared-trunk parameter, for per-block logging."""
    if name.startswith("backbone.bottom_up.stem"):
        return "stem"
    for r in ("res2", "res3", "res4", "res5"):
        if name.startswith(f"backbone.bottom_up.{r}"):
            return r
    return "fpn"


def _cagrad_weight(g11: float, g12: float, g22: float, alpha: float) -> float:
    """CAGrad (Liu et al., NeurIPS 2021) mixing weight for the two-task case.

    Minimises F(x) = <g_x, g0> + phi * ||g_x|| over x in [0, 1], where
    g_x = x*g1 + (1-x)*g2, g0 = (g1+g2)/2 and phi = alpha*||g0||. The official
    implementation calls scipy's SLSQP; with two tasks the search is a single
    bounded scalar, so a dense grid plus a local refinement is exact enough and
    keeps the run dependency-free and deterministic.
    """
    g0_sq = 0.25 * (g11 + 2.0 * g12 + g22)
    phi = alpha * math.sqrt(max(g0_sq, 0.0))

    def F(x: float) -> float:
        gx_g0 = 0.5 * (x * (g11 + g12) + (1.0 - x) * (g12 + g22))
        gx_sq = x * x * g11 + 2.0 * x * (1.0 - x) * g12 + (1.0 - x) * (1.0 - x) * g22
        return gx_g0 + phi * math.sqrt(max(gx_sq, 0.0))

    xs = [i / 200.0 for i in range(201)]
    best = min(xs, key=F)
    lo, hi = max(0.0, best - 0.005), min(1.0, best + 0.005)
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if F(m1) < F(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


# =====================================================================
# CT-CMT-MTL meta-arch
# =====================================================================
@META_ARCH_REGISTRY.register()
class CTCMT_MTL(nn.Module):
    """Cross-Task Consistent Mean Teacher for detectron2 PanopticFPN."""

    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        anchor: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.optimizer = optimizer

        self.num_classes = int(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
        # MR-CNN configs have no SEM_SEG_HEAD; keep num_seg_classes optional.
        _seg_head_cfg = getattr(cfg.MODEL, "SEM_SEG_HEAD", None)
        self.num_seg_classes = int(_seg_head_cfg.NUM_CLASSES) if _seg_head_cfg is not None else 0

        # Mean-teacher det hyperparams (mirrors AMROD).
        self.threshold_init = float(cfg.SOLVER.THRESHOLD_INIT)
        self.thresholds_max = float(cfg.SOLVER.THRESHOLD_MAX)
        self.thresholds_mini = float(cfg.SOLVER.THRESHOLD_MINI)
        self.alpha_dt = float(cfg.SOLVER.ALPHA_DT)
        self.gamma_dt = float(cfg.SOLVER.GAMMA_DT)
        self.thresholds = [self.threshold_init] * self.num_classes

        # Score-EM gating (skip step when teacher confidence is stable).
        self.score_em = float(cfg.SOLVER.SCORE_EM)
        self.score_gamma = float(cfg.SOLVER.SCORE_GAMMA)
        self.score_thresh = float(cfg.SOLVER.SCORE_THRESH)
        # When set, ignore the gate and let every image contribute a step.
        self.skip_score_em_gate = bool(getattr(cfg.SOLVER, "CTCMT_SKIP_SCORE_EM_GATE", False))
        self.predict_before_update = bool(getattr(cfg.SOLVER, "CTCMT_PREDICT_BEFORE_UPDATE", False))

        # Loss weights.
        self.weight_det = float(cfg.SOLVER.CTCMT_WEIGHT_DET)
        self.weight_seg = float(cfg.SOLVER.CTCMT_WEIGHT_SEG)
        self.weight_ctcl = float(cfg.SOLVER.CTCMT_WEIGHT_CTCL)
        # Cross-task consistency regularizer: pixels inside a teacher-detected
        # bbox should be classified by the student seg head as that box's class.
        self.weight_ctcr = float(getattr(cfg.SOLVER, "CTCMT_WEIGHT_CTCR", 0.0))

        # CT-CR spatial supervision mode.
        self.ctcr_mode = str(
            getattr(cfg.SOLVER, "CTCMT_CTCR_MODE", "full_box")
        ).lower()
        self.ctcr_mask_thresh = float(
            getattr(cfg.SOLVER, "CTCMT_CTCR_MASK_THRESH", 0.3)
        )
        self.ctcr_weight_floor = float(
            getattr(cfg.SOLVER, "CTCMT_CTCR_WEIGHT_FLOOR", 0.2)
        )

        _valid_ctcr_modes = {
            "full_box", "per_box_full", "hard_seg", "soft_seg", "soft_seg_global",
        }
        if self.ctcr_mode not in _valid_ctcr_modes:
            raise ValueError(
                f"Unsupported CTCMT_CTCR_MODE={self.ctcr_mode!r}; "
                f"expected one of {sorted(_valid_ctcr_modes)}"
            )
        if not 0.0 <= self.ctcr_mask_thresh <= 1.0:
            raise ValueError(
                "CTCMT_CTCR_MASK_THRESH must be in [0, 1], got "
                f"{self.ctcr_mask_thresh}"
            )
        if not 0.0 <= self.ctcr_weight_floor <= 1.0:
            raise ValueError(
                "CTCMT_CTCR_WEIGHT_FLOOR must be in [0, 1], got "
                f"{self.ctcr_weight_floor}"
            )

        # Detector-dominant CT-CR. Both default off and leave the loss
        # bit-identical; see defaults.py for the measurements behind them.
        self.ctcr_score_order = bool(
            getattr(cfg.SOLVER, "CTCMT_CTCR_SCORE_ORDER", False)
        )
        self.ctcr_det_score_weight = bool(
            getattr(cfg.SOLVER, "CTCMT_CTCR_DET_SCORE_WEIGHT", False)
        )
        self.ctcr_det_score_gamma = float(
            getattr(cfg.SOLVER, "CTCMT_CTCR_DET_SCORE_GAMMA", 1.0)
        )

        # Inverted CTPV: a confident detection suppresses the segmentation
        # soft-CE where the seg teacher disagrees with the mapped class.
        self.seg_det_veto = bool(getattr(cfg.SOLVER, "CTCMT_SEG_DET_VETO", False))
        self.seg_det_veto_score = float(
            getattr(cfg.SOLVER, "CTCMT_SEG_DET_VETO_SCORE", 0.9)
        )
        self.seg_det_veto_weight = float(
            getattr(cfg.SOLVER, "CTCMT_SEG_DET_VETO_WEIGHT", 0.0)
        )

        # Single-task ablation switches: disable one task branch entirely so
        # this meta-arch degenerates to a fair single-task-on-MTL-source baseline.
        self.det_only = bool(getattr(cfg.SOLVER, "CTCMT_DET_ONLY", False))
        self.seg_only = bool(getattr(cfg.SOLVER, "CTCMT_SEG_ONLY", False))
        # CoTTA-style multi-scale augmentation-averaged seg pseudo-labels.
        self.seg_aug_enabled = bool(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_ENABLED", False))
        self.seg_aug_conf_thresh = float(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_CONF_THRESH", 0.9))
        self.seg_aug_scales = tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_SCALES", (1.0,)))
        self.seg_aug_flips = tuple(getattr(cfg.SOLVER, "CTCMT_SEG_AUG_FLIPS", (False,)))

        # Cross-task contrastive.
        self.ctcl_enabled = bool(cfg.SOLVER.CTCMT_CTCL_ENABLED)
        self.ctcl_include_seg_view = bool(cfg.SOLVER.CTCMT_CTCL_SEG_VIEW)
        self.ctcl_temperature = float(cfg.SOLVER.CTCMT_CTCL_TEMPERATURE)
        self.ctcl_roi_output = tuple(cfg.SOLVER.CTCMT_CTCL_ROI_OUTPUT)
        self.ctcl_proj_dim = int(cfg.SOLVER.CTCMT_CTCL_PROJ_DIM)

        # CoTTA-style EMA + stochastic restore.
        self.ema_decay = float(cfg.SOLVER.MT)
        self.restore_prob = float(cfg.SOLVER.RST_M)

        # --- Novel extension V1: per-task decoupled adaptation gates ---
        # Instead of one global gate, each task branch fires independently when
        # its own confidence is low (= domain shift detected for that task).
        self.per_task_gate = bool(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE", False))
        self.per_task_gate_det_thresh = float(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE_DET_THRESH", 0.8))
        self.per_task_gate_seg_thresh = float(getattr(cfg.SOLVER, "CTCMT_PER_TASK_GATE_SEG_THRESH", 0.8))

        # --- Novel extension V2: task-aware stochastic restore ---
        # Shared backbone / FPN params are used by both tasks; restore them
        # with a lower probability than task-specific head params.
        self.cross_task_fisher = bool(getattr(cfg.SOLVER, "CTCMT_CROSS_TASK_FISHER", False))
        # backbone restore rate = restore_prob * this factor (< 1 = more protection)
        self.backbone_rst_factor = float(getattr(cfg.SOLVER, "CTCMT_BACKBONE_RST_FACTOR", 0.1))

        # Weak/strong mean-teacher asymmetry (AMROD-style): the teacher keeps the
        # weak view, the student is trained on "image_strong".
        self.strong_aug_student = bool(getattr(cfg.SOLVER, "CTCMT_STRONG_AUG_STUDENT", False))

        # DIAGNOSTIC ONLY. This is AMROD's "Randomized Restoration" (Wei et al.,
        # one of that paper's two titular contributions), ported to quantify how
        # much of its advantage comes from restoration. Never report as ours.
        self.fisher_restore = bool(getattr(cfg.SOLVER, "CTCMT_FISHER_RESTORE", False))

        # Counteract minority-class collapse: the plain pixel-mean soft-CE is
        # dominated by frequent classes, so self-distillation absorbs rare
        # classes into their dominant neighbours over long streams.
        self.class_balanced_ce = bool(getattr(cfg.SOLVER, "CTCMT_CLASS_BALANCED_CE", False))
        self.class_balance_beta = float(getattr(cfg.SOLVER, "CTCMT_CLASS_BALANCE_BETA", 0.5))
        self.class_marginal_ema = float(getattr(cfg.SOLVER, "CTCMT_CLASS_MARGINAL_EMA", 0.999))
        self._class_marginal = None
        # Reweighting raises the seg loss magnitude (rarity correlates with error);
        # this keeps det/seg gradient balance fixed so the ablation stays clean.
        self.seg_scale_preserve = bool(
            getattr(cfg.SOLVER, "CTCMT_SEG_LOSS_SCALE_PRESERVE", False)
        )

        self.anchor_marginal_weight = float(
            getattr(cfg.SOLVER, "CTCMT_ANCHOR_MARGINAL_WEIGHT", 0.0)
        )

        # --- Novel extension V3: cross-task pseudo-label verification ---
        # Reject a det box as pseudo-label if the seg head disagrees with its class.
        self.ctpv_enabled = bool(getattr(cfg.SOLVER, "CTCMT_CTPV_ENABLED", False))
        self.ctpv_thresh = float(getattr(cfg.SOLVER, "CTCMT_CTPV_THRESH", 0.3))

        # --- Novel extension V4: cross-task prototype anchor ---
        # Running EMA prototypes (det-view + seg-view) per class. Updated only
        # when both views agree. Add a weak pull toward stored prototypes.
        self.proto_anchor = bool(getattr(cfg.SOLVER, "CTCMT_PROTO_ANCHOR", False))
        self.proto_ema = float(getattr(cfg.SOLVER, "CTCMT_PROTO_EMA", 0.999))
        self.proto_weight = float(getattr(cfg.SOLVER, "CTCMT_PROTO_WEIGHT", 0.01))
        self._det_protos: Dict[int, torch.Tensor] = {}   # class -> (D,) det-view
        self._seg_protos: Dict[int, torch.Tensor] = {}   # class -> (D,) seg-view

        # --- Enhancement E2: entropy-weighted soft-CE (down-weight uncertain pixels) ---
        self.entropy_weighted_ce = bool(getattr(cfg.SOLVER, "CTCMT_ENTROPY_WEIGHTED_CE", False))

        # --- Enhancement E3: trigger seg aug-avg on TEACHER entropy, not anchor confidence ---
        self.aug_trigger_teacher_entropy = bool(getattr(cfg.SOLVER, "CTCMT_AUG_TRIGGER_TEACHER_ENTROPY", False))
        self.aug_teacher_entropy_thresh = float(getattr(cfg.SOLVER, "CTCMT_AUG_TEACHER_ENTROPY_THRESH", 0.3))

        # --- Enhancement E4: directional score-EM gate (skip on stable, adapt on shift) ---
        self.directional_gate = bool(getattr(cfg.SOLVER, "CTCMT_DIRECTIONAL_GATE", False))
        self.dir_gate_stable_band = float(getattr(cfg.SOLVER, "CTCMT_DIR_GATE_STABLE_BAND", 0.4))
        self.dir_gate_boost = float(getattr(cfg.SOLVER, "CTCMT_DIR_GATE_BOOST", 2.0))
        # Boost applied to CT-CL/CT-CR when a downward shift is detected.
        self._cl_boost = 1.0

        # --- Enhancement E5: adaptive STR (η depends on measured shared-trunk drift) ---
        self.adaptive_str = bool(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR", False))
        self.adaptive_str_base = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_BASE", 0.1))
        self.adaptive_str_boost = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_BOOST", 0.4))
        self.adaptive_str_pivot = float(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_STR_PIVOT", 0.05))
        self._backbone_source_norm = None   # cached lazily after _source_params snapshot

        # --- Negative-transfer screening batch (S1..S5) ---
        self.conflict_mode = str(
            getattr(cfg.SOLVER, "CTCMT_CONFLICT_MODE", "none")
        ).lower()
        if self.conflict_mode not in _VALID_CONFLICT_MODES:
            raise ValueError(
                f"Unsupported CTCMT_CONFLICT_MODE={self.conflict_mode!r}; "
                f"expected one of {sorted(_VALID_CONFLICT_MODES)}"
            )
        self.cagrad_alpha = float(getattr(cfg.SOLVER, "CTCMT_CAGRAD_ALPHA", 0.5))

        # Task-CoCo: official CoCo-MT-TTA Gradient Consensus defaults.
        # Adapted from leafheavy/MT-TTA/GradientConsensus.py (MIT License,
        # Copyright (c) 2025 leafheavy).  These defaults intentionally mirror
        # the official implementation: GC_c=0.5, 100 Adam iterations, lr=1.0,
        # eps=1e-4, Adam betas=(0.9,0.999), weight_decay=1e-4.
        #
        # Only the Gradient Consensus component is used here.  CoCo's separate
        # Fisher/Plasticity Constraint is NOT enabled in this ablation.
        self.task_coco_gc_c = float(
            getattr(cfg.SOLVER, "CTCMT_TASK_COCO_GC_C", 0.5)
        )
        self.task_coco_iters = max(int(
            getattr(cfg.SOLVER, "CTCMT_TASK_COCO_ITERS", 100)
        ), 1)
        self.task_coco_lr = float(
            getattr(cfg.SOLVER, "CTCMT_TASK_COCO_LR", 1.0)
        )
        self.task_coco_eps = float(
            getattr(cfg.SOLVER, "CTCMT_TASK_COCO_EPS", 1e-4)
        )

        self.freeze_shared_trunk = bool(
            getattr(cfg.SOLVER, "CTCMT_FREEZE_SHARED_TRUNK", False)
        )
        self.aux_trunk_lambda = float(getattr(cfg.SOLVER, "CTCMT_AUX_TRUNK_LAMBDA", 0.0))
        if not 0.0 <= self.aux_trunk_lambda <= 1.0:
            raise ValueError(
                "CTCMT_AUX_TRUNK_LAMBDA must be in [0, 1], got "
                f"{self.aux_trunk_lambda}"
            )
        self.adaptive_routing = bool(getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_ROUTING", False))
        self.adaptive_routing_beta = float(
            getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_ROUTING_BETA", 0.98))
        self.adaptive_routing_ema = float(
            getattr(cfg.SOLVER, "CTCMT_ADAPTIVE_ROUTING_EMA", 0.99))
        self._seg_agree_ema = None      # teacher/anchor pixel agreement
        self._seg_agree_max = 0.0
        self._route_lambda = self.aux_trunk_lambda
        self.route_scope = str(
            getattr(cfg.SOLVER, "CTCMT_ROUTE_SCOPE", "shared")
        ).lower()
        if self.route_scope not in ("shared", "fpn"):
            raise ValueError(
                f"Unsupported CTCMT_ROUTE_SCOPE={self.route_scope!r}; "
                "expected 'shared' or 'fpn'"
            )
        self.grad_diag = bool(getattr(cfg.SOLVER, "CTCMT_GRAD_DIAG", False))
        self.grad_diag_every = max(int(getattr(cfg.SOLVER, "CTCMT_GRAD_DIAG_EVERY", 50)), 1)
        self._param_index = None            # [(name, param)], built lazily
        self._shared_names = None           # names in ResNet + FPN
        self._shared_set = None
        self._resnet_names = None           # ResNet bottom-up only
        self._resnet_set = None
        self._fpn_names = None              # shared backbone minus ResNet bottom-up
        self._fpn_set = None
        self._routed_set = None             # legacy aux_head_only route-scope subset
        self._conflict_stats = {
            "steps": 0, "both": 0, "conflicts": 0, "projected": 0,
            "cos_sum": 0.0, "gamma_sum": 0.0, "w_det_sum": 0.0,
            "w_seg_sum": 0.0, "c_det_sum": 0.0, "c_seg_sum": 0.0,
        }
        self._grad_fallbacks = 0

        # --- Experimental Fisher variants (paper26_fisher_*), default OFF ---
        self.fisher_guided_restore = bool(
            getattr(cfg.SOLVER, "CTCMT_FISHER_RESTORE_ENABLED", False))
        self.fisher_xtask = bool(getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_ENABLED", False))
        self.fx_beta = float(getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_BETA", 0.99))
        self.fx_warmup = int(getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_WARMUP", 50))
        self.fx_lambda = float(getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_LAMBDA", 0.5))
        self.fx_scope = str(getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_SCOPE", "resnet")).lower()
        self.fx_gran = str(
            getattr(cfg.SOLVER, "CTCMT_FISHER_XTASK_GRANULARITY", "param")).lower()
        if self.fisher_guided_restore and self.fisher_restore:
            raise ValueError("CTCMT_FISHER_RESTORE_ENABLED and the legacy "
                             "CTCMT_FISHER_RESTORE are mutually exclusive")
        if self.fisher_guided_restore and self.adaptive_str:
            raise ValueError("CTCMT_FISHER_RESTORE_ENABLED does not support ADAPTIVE_STR")
        if self.fisher_xtask:
            if self.conflict_mode != "none":
                raise ValueError("CTCMT_FISHER_XTASK_ENABLED requires "
                                 "CTCMT_CONFLICT_MODE='none'")
            if self.fx_scope != "resnet":
                raise NotImplementedError("CTCMT_FISHER_XTASK_SCOPE: only 'resnet'")
            if self.fx_gran != "param":
                raise NotImplementedError("CTCMT_FISHER_XTASK_GRANULARITY: only 'param'")
            if not 0.0 <= self.fx_beta < 1.0 or not 0.0 <= self.fx_lambda <= 1.0:
                raise ValueError("bad CTCMT_FISHER_XTASK_BETA / _LAMBDA")
        self._fx_F_det = None        # name -> FP32 EMA of g_det^2 (ResNet only)
        self._fx_F_aux = None        # name -> FP32 EMA of g_aux^2
        self._fx_n_active = 0        # detection-active (paired) updates so far
        self._fx_stats = None        # [3 blocks, 11] GPU accumulator, flushed in logs
        self._fx_stat_steps = 0
        self._fr_has_grad = False
        self._fr_stats = None        # GPU accumulator for Fisher-restore sanity stats
        # How often the conditional CoTTA-style aug-averaging actually fired.
        self._seg_aug_fired = 0
        self._seg_steps = 0

        self.iter = 0
        self._fx_out_dir = str(cfg.OUTPUT_DIR)
        # ---------------------------------------------------------------
        # Optional diagnostics.
        #
        # Pure logging only: does not affect adaptation behavior.
        # Enable with:
        #
        #   CTCMT_DIAG_JSONL=/workspace/output/.../ctpv_trace.jsonl
        #
        # ---------------------------------------------------------------
        self.diag_jsonl = os.environ.get("CTCMT_DIAG_JSONL", "").strip()
        self._diag_raw_teacher_n = 0
        self._diag_after_dyn_n = 0

        # Optional diagnostic-only semantic GT access for masked CT-CR analysis.
        # IMPORTANT: GT is used only to compute logging statistics. It is never
        # used by a loss, gate, pseudo-label filter, optimizer step, EMA update,
        # or stochastic restoration.
        self.mask_diag_gt_root = os.environ.get("CTCMT_MASK_DIAG_GT_ROOT", "").strip()
        self.mask_diag_thresholds = tuple(
            float(x) for x in os.environ.get(
                "CTCMT_MASK_DIAG_THRESHOLDS", "0.1,0.2,0.3,0.5,0.7,0.9"
            ).split(",") if x.strip()
        )
        self._mask_diag_gt_index = {}
        if self.mask_diag_gt_root:
            try:
                for dirpath, _, filenames in os.walk(self.mask_diag_gt_root):
                    for fn in filenames:
                        suffix = "_gtFine_labelTrainIds.png"
                        if fn.endswith(suffix):
                            key = fn[:-len(suffix)]
                            self._mask_diag_gt_index[key] = os.path.join(dirpath, fn)
            except Exception as exc:
                print(f"[CT-CMT-DIAG] failed to index semantic GT: {exc}")
                self._mask_diag_gt_index = {}

        # Snapshot every trainable weight/bias in the anchor for stochastic restore.
        self._source_params: Dict[str, torch.Tensor] = {}
        for nm, m in self.anchor.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name in ("weight", "bias"):
                    self._source_params[f"{nm}.{np_name}"] = p.detach().clone()

        # Contrastive projection heads (built lazily on first CTCL call once
        # we know the FPN channel count).
        self._proj_det = None
        self._proj_seg = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg):
        # Student meta-arch is configurable so CTCMT_MTL can wrap either a
        # PanopticFPN (MTL / seg-only / det-only on PFN source) or a
        # GeneralizedRCNN (det-only on a Mask R-CNN source).
        student_arch_name = getattr(cfg.MODEL, "CTCMT_STUDENT_META_ARCH", "PanopticFPN")
        pfn_cls = META_ARCH_REGISTRY.get(student_arch_name)
        weights_path = cfg.MODEL.WEIGHTS

        def _build_and_load(train_mode: bool, freeze: bool, disable_mask_head: bool = True):
            m = pfn_cls(cfg)
            # Use .load() (not resume_or_load) so FrozenBN buffers reliably
            # come from the checkpoint.
            DetectionCheckpointer(m).load(weights_path)
            m.to(torch.device(cfg.MODEL.DEVICE))
            if train_mode:
                m.train()
            else:
                m.eval()
            if freeze:
                for p in m.parameters():
                    p.requires_grad_(False)
            # No pseudo-masks -> disable mask head on student/anchor to skip
            # wasteful mask computation during adaptation. Keep it on the
            # teacher so the evaluator receives pred_masks (e.g. cityscapes).
            # A SemanticSegmentor student has no roi_heads at all.
            if (disable_mask_head and hasattr(m, "roi_heads")
                    and hasattr(m.roi_heads, "mask_on")):
                m.roi_heads.mask_on = False
            return m

        student = _build_and_load(train_mode=True,  freeze=False, disable_mask_head=True)
        teacher = _build_and_load(train_mode=True,  freeze=True,  disable_mask_head=False)
        anchor  = _build_and_load(train_mode=False, freeze=True,  disable_mask_head=True)

        # S4: structural parameter isolation. Must happen BEFORE build_optimizer
        # so the frozen tensors never enter a param group (and so momentum /
        # weight decay can never touch them).
        if bool(getattr(cfg.SOLVER, "CTCMT_FREEZE_SHARED_TRUNK", False)):
            n_frozen = n_train = 0
            for nm, p in student.named_parameters():
                if any(nm.startswith(pfx) for pfx in _SHARED_PARAM_PREFIXES):
                    p.requires_grad_(False)
                    n_frozen += p.numel()
                elif p.requires_grad:
                    n_train += p.numel()
            trainable = sorted({
                nm.split(".")[0] for nm, p in student.named_parameters() if p.requires_grad
            })
            print(f"[CT-CMT-MTL] FREEZE_SHARED_TRUNK: frozen={n_frozen/1e6:.2f}M "
                  f"trainable={n_train/1e6:.2f}M modules={trainable}")

        optimizer = build_optimizer(cfg, student)
        return {
            "student": student,
            "teacher": teacher,
            "anchor": anchor,
            "optimizer": optimizer,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.student.pixel_mean.device

    # ------------------------------------------------------------------
    # Teacher pseudo-labels via dynamic per-class thresholds + score-EM gate.
    # Returns (list[Instances], keep_step: bool, score_summary: str)
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_instances(images):
        """Zero-length Instances carrying the fields the rest of forward() reads.

        A bare ``Instances()`` raises on ``__len__``, and the periodic log line
        reads ``len(pseudo_inst[0])`` unconditionally.
        """
        dev = images.tensor.device
        inst = Instances(tuple(images.image_sizes[0]))
        inst.pred_boxes = Boxes(torch.zeros((0, 4), device=dev))
        inst.pred_classes = torch.zeros((0,), dtype=torch.long, device=dev)
        inst.scores = torch.zeros((0,), device=dev)
        return inst

    @torch.no_grad()
    def _teacher_pseudo(self, batched_inputs):
        if self.seg_only:
            # Segmentation-only student (e.g. SemanticSegmentor): no detector
            # exists, so take the teacher's seg logits directly on the weak view
            # and hand back an empty-but-well-formed pseudo-instance set.
            images = self.teacher.preprocess_image(batched_inputs)
            sem_seg_results, _ = self.teacher.sem_seg_head(
                self.teacher.backbone(images.tensor), None
            )
            self._diag_raw_teacher_n = 0
            self._diag_after_dyn_n = 0
            return [self._empty_instances(images)], sem_seg_results, False

        # Raw teacher outputs (pre-postprocess), in the RESIZED image frame.
        # PanopticFPN returns (det, sem); GeneralizedRCNN returns det only.
        teacher_out = self.teacher.inference(batched_inputs, do_postprocess=False)
        if isinstance(teacher_out, tuple) and len(teacher_out) == 2:
            detector_results, sem_seg_results = teacher_out
        else:
            detector_results, sem_seg_results = teacher_out, None
        inst = detector_results[0]
        self._diag_raw_teacher_n = int(len(inst))
        if len(inst) == 0:
            self._diag_after_dyn_n = 0
            return detector_results, sem_seg_results, False

        # Score-EM gate: skip step if teacher confidence is stable.
        valid_mask = inst.scores > 0.1
        if not valid_mask.any():
            # No dynamic-threshold filtering is applied on this early-return path,
            # so the returned pseudo set is the raw detector result.
            self._diag_after_dyn_n = int(len(detector_results[0]))
            return detector_results, sem_seg_results, False
        mean_all = float(inst.scores[valid_mask].mean().cpu())
        keep_step = True
        if self.score_em > 0 and not self.skip_score_em_gate:
            ratio = mean_all / self.score_em
            if self.directional_gate:
                # E4: skip only when high AND stable; force adapt (and boost CT-CL) on shift down.
                if mean_all >= self.score_em and abs(ratio - 1.0) < self.dir_gate_stable_band:
                    keep_step = False   # already adapted
                    self._cl_boost = 1.0
                elif ratio < 0.6:
                    self._cl_boost = self.dir_gate_boost   # sharp drop = shift onset
                else:
                    self._cl_boost = 1.0
            else:
                if ratio > self.score_thresh or (1.0 / max(ratio, 1e-6)) > self.score_thresh:
                    keep_step = False
        self.score_em = self.score_gamma * self.score_em + (1.0 - self.score_gamma) * mean_all

        # Per-class mean score -> new dynamic thresholds.
        per_class_mean = [0.0] * self.num_classes
        classes = inst.pred_classes[valid_mask]
        scores = inst.scores[valid_mask]
        for c in range(self.num_classes):
            idx = classes == c
            if int(idx.sum()) > 0:
                per_class_mean[c] = float(scores[idx].mean())
        self.thresholds = _dyn_thresholds(
            self.thresholds, per_class_mean,
            self.alpha_dt, self.gamma_dt,
            self.thresholds_mini, self.thresholds_max,
        )

        # Filter with dyn thresholds.
        thr = torch.tensor(self.thresholds, device=inst.scores.device)
        keep = inst.scores >= thr[inst.pred_classes.long()]
        filtered = Instances(inst.image_size)
        filtered.pred_boxes = Boxes(inst.pred_boxes.tensor[keep])
        filtered.pred_classes = inst.pred_classes[keep]
        filtered.scores = inst.scores[keep]
        for k in inst.get_fields():
            if k not in ("pred_boxes", "pred_classes", "scores"):
                filtered.set(k, inst.get(k)[keep])
        self._diag_after_dyn_n = int(len(filtered))
        return [filtered], sem_seg_results, keep_step

    # ------------------------------------------------------------------
    # Convert filtered teacher instances -> pseudo gt_instances the student
    # detection heads expect (uses gt_boxes / gt_classes).
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gt_instances(instances_list):
        out = []
        for inst in instances_list:
            g = Instances(inst.image_size)
            g.gt_boxes = Boxes(inst.pred_boxes.tensor.clone())
            g.gt_classes = inst.pred_classes.long().clone()
            out.append(g)
        return out

    # ------------------------------------------------------------------
    # Cross-task contrastive on student FPN features.
    # ------------------------------------------------------------------
    def _ctcl_loss(self, student_features, pseudo_instances, teacher_sem_probs):
        inst = pseudo_instances[0]
        if len(inst) == 0:
            return student_features[next(iter(student_features))].new_zeros(())

        # Take the deepest FPN level that's in ROI box features for a compact view.
        feat_key = self.student.roi_heads.box_in_features[-1]
        feat = student_features[feat_key]  # (B, C, H, W)
        C = feat.size(1)

        # Convert boxes from RESIZED input coords to feat coords via FPN stride.
        stride = self.student.backbone.output_shape()[feat_key].stride
        boxes = inst.pred_boxes.tensor / float(stride)
        classes = inst.pred_classes.long()

        # ---- Det view: RoI-align pooled features -> mean over spatial -> L2 norm.
        # Raw features (no random-init projection heads) match the shift-tta reference.
        rois = torch.cat(
            [torch.zeros((boxes.size(0), 1), device=boxes.device), boxes], dim=1
        )
        pooled = roi_align(feat, rois, output_size=self.ctcl_roi_output,
                           spatial_scale=1.0, aligned=True)  # (N, C, h, w)
        z_det = pooled.mean(dim=(2, 3))                         # (N, C)
        z_det = F.normalize(z_det, dim=1)

        views = [z_det]
        labels = [classes]

        if self.ctcl_include_seg_view and teacher_sem_probs is not None:
            # Downsample teacher probs to feat resolution.
            probs_feat = F.interpolate(
                teacher_sem_probs, size=feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            H, W = feat.shape[-2:]
            z_seg_list = []
            for b, c in zip(boxes.detach().cpu().tolist(), classes.detach().cpu().tolist()):
                # Map detection class -> seg channel (Cityscapes 8->19 taxonomy).
                if 0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES):
                    seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
                else:
                    continue
                if seg_c >= self.num_seg_classes:
                    continue
                x1, y1, x2, y2 = [max(int(round(v)), 0) for v in b]
                x2 = min(x2, W)
                y2 = min(y2, H)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_f = feat[0:1, :, y1:y2, x1:x2]                     # (1, C, h, w)
                crop_p = probs_feat[0:1, seg_c:seg_c + 1, y1:y2, x1:x2] # (1, 1, h, w)
                w_sum = crop_p.sum().clamp(min=1e-6)
                pooled_seg = (crop_f * crop_p).sum(dim=(2, 3)) / w_sum  # (1, C)
                # Label with the DETECTION class so SupCon pulls det-view and
                # seg-view of the same object together (cross-task pairs).
                z_seg_list.append((pooled_seg.squeeze(0), c))
            if z_seg_list:
                z_seg = torch.stack([z for z, _ in z_seg_list], dim=0)
                z_seg = F.normalize(z_seg, dim=1)
                seg_classes = torch.tensor(
                    [c for _, c in z_seg_list], device=z_seg.device, dtype=classes.dtype
                )
                views.append(z_seg)
                labels.append(seg_classes)

        z = torch.cat(views, dim=0)
        y = torch.cat(labels, dim=0)
        return _supcon_loss(z, y, temperature=self.ctcl_temperature)

    # ------------------------------------------------------------------
    # Cross-task consistency regularizer: student seg logits inside teacher
    # boxes must classify as that box's (seg-taxonomy) class.
    # ------------------------------------------------------------------
    def _ctcr_loss(
        self,
        s_seg_logits,
        pseudo_instances,
        teacher_seg_probs=None,
    ):
        """Cross-task consistency regularizer with selectable spatial mode."""
        inst = pseudo_instances[0]
        if len(inst) == 0 or s_seg_logits is None:
            return None

        # A / D) Global target-map path.
        #
        # "full_box" is the legacy A formulation: uniform weight, cross-entropy
        # normalized per supervised PIXEL.
        #
        # "soft_seg_global" is the D cell of the CT-CR 2x2: identical target-map
        # construction and overlap-overwrite order as A, but each supervised
        # pixel is weighted by w = floor + (1 - floor) * q, with q the detached
        # teacher posterior for the mapped semantic class. Normalization stays
        # global (sum w) rather than per box, so D isolates soft semantic
        # weighting from the per-box aggregation change made by A2/B/C.
        # With floor = 1.0, D reduces exactly to A.
        if self.ctcr_mode in ("full_box", "soft_seg_global"):
            use_soft_global = self.ctcr_mode == "soft_seg_global"
            use_weights = use_soft_global or self.ctcr_det_score_weight

            B, K, H, W = s_seg_logits.shape
            target = torch.full(
                (B, H, W), 255, dtype=torch.long, device=s_seg_logits.device
            )

            probs = None
            weight = None
            if use_soft_global:
                if teacher_seg_probs is None:
                    return None
                probs = teacher_seg_probs.detach()
                if probs.shape[-2:] != (H, W):
                    probs = F.interpolate(
                        probs.float(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
            if use_weights:
                weight = torch.zeros(
                    (B, H, W), dtype=torch.float32, device=s_seg_logits.device
                )

            img_h, img_w = inst.image_size
            sx = W / max(img_w, 1)
            sy = H / max(img_h, 1)
            boxes = inst.pred_boxes.tensor.detach()
            classes = inst.pred_classes.detach().long().tolist()
            box_list = boxes.tolist()
            scores = (
                inst.scores.detach().tolist() if inst.has("scores")
                else [1.0] * len(classes)
            )

            order = list(range(len(classes)))
            if self.ctcr_score_order:
                # Later boxes overwrite earlier ones, and detections arrive
                # score-descending, so ascending order lets the best box win.
                order.sort(key=lambda i: scores[i])

            for j in order:
                x1, y1, x2, y2 = box_list[j]
                c = classes[j]
                if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                    continue
                seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
                if seg_c >= K:
                    continue
                if probs is not None and seg_c >= probs.shape[1]:
                    continue

                x1i = max(int(round(x1 * sx)), 0)
                y1i = max(int(round(y1 * sy)), 0)
                x2i = min(int(round(x2 * sx)), W)
                y2i = min(int(round(y2 * sy)), H)

                if x2i <= x1i or y2i <= y1i:
                    continue

                target[0, y1i:y2i, x1i:x2i] = seg_c

                if use_weights:
                    if use_soft_global:
                        q = probs[0, seg_c, y1i:y2i, x1i:x2i].float().clamp(0.0, 1.0)
                        floor = self.ctcr_weight_floor
                        w_box = floor + (1.0 - floor) * q
                    else:
                        w_box = torch.ones(
                            (y2i - y1i, x2i - x1i),
                            dtype=torch.float32,
                            device=s_seg_logits.device,
                        )
                    if self.ctcr_det_score_weight:
                        w_box = w_box * (
                            max(float(scores[j]), 0.0) ** self.ctcr_det_score_gamma
                        )
                    weight[0, y1i:y2i, x1i:x2i] = w_box

            valid = target != 255
            if int(valid.sum()) == 0:
                return None

            if not use_weights:
                return F.cross_entropy(
                    s_seg_logits, target, ignore_index=255
                )

            ce = F.cross_entropy(
                s_seg_logits, target, ignore_index=255, reduction="none"
            )
            w = weight * valid.float()
            return (w * ce).sum() / w.sum().clamp_min(1e-6)

        # A2/B/C) Per-box CT-CR modes.
        #
        # "per_box_full" is the controlled A2 ablation:
        #   - full rectangular bbox supervision (same semantic assumption as A)
        #   - NO teacher-semantic mask or weighting
        #   - per-box CE normalization + equal mean across boxes (same aggregation
        #     machinery as B/C)
        #
        # This separates the effect of spatial semantic reliability from the
        # legacy global target-map / overlap-overwrite aggregation used by A.
        B, K, H, W = s_seg_logits.shape
        probs = None

        if self.ctcr_mode != "per_box_full":
            if teacher_seg_probs is None:
                return None

            probs = teacher_seg_probs.detach()
            if probs.shape[-2:] != (H, W):
                probs = F.interpolate(
                    probs.float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )

        img_h, img_w = inst.image_size
        sx = W / max(img_w, 1)
        sy = H / max(img_h, 1)

        boxes = inst.pred_boxes.tensor.detach()
        classes = inst.pred_classes.detach().long().tolist()
        box_scores = (
            inst.scores.detach().tolist() if inst.has("scores")
            else [1.0] * len(classes)
        )
        box_losses = []
        box_weights = []

        for j, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
            c = classes[j]
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                continue

            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K:
                continue
            if probs is not None and seg_c >= probs.shape[1]:
                continue

            x1i = max(int(round(x1 * sx)), 0)
            y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W)
            y2i = min(int(round(y2 * sy)), H)

            if x2i <= x1i or y2i <= y1i:
                continue

            logits_crop = s_seg_logits[
                0:1, :, y1i:y2i, x1i:x2i
            ].float()

            h = y2i - y1i
            w = x2i - x1i
            target_crop = torch.full(
                (1, h, w),
                int(seg_c),
                dtype=torch.long,
                device=s_seg_logits.device,
            )

            ce = F.cross_entropy(
                logits_crop,
                target_crop,
                reduction="none",
            )[0]

            if self.ctcr_mode == "per_box_full":
                # A2 control: every pixel in the rectangle is supervised
                # uniformly, but each box is normalized independently.
                box_loss = ce.mean()

            else:
                q = probs[
                    0, seg_c, y1i:y2i, x1i:x2i
                ].float().clamp(0.0, 1.0)

                if self.ctcr_mode == "hard_seg":
                    mask = q >= self.ctcr_mask_thresh
                    if not bool(mask.any()):
                        continue
                    box_loss = ce[mask].mean()

                elif self.ctcr_mode == "soft_seg":
                    floor = self.ctcr_weight_floor
                    weights = floor + (1.0 - floor) * q
                    box_loss = (
                        (weights * ce).sum()
                        / weights.sum().clamp_min(1e-6)
                    )

                else:
                    raise RuntimeError(
                        f"Unexpected CT-CR mode: {self.ctcr_mode}"
                    )

            box_losses.append(box_loss)
            box_weights.append(
                max(float(box_scores[j]), 0.0) ** self.ctcr_det_score_gamma
            )

        if not box_losses:
            return None

        if not self.ctcr_det_score_weight:
            # Equal box contribution; no det-score weighting in this ablation.
            return torch.stack(box_losses).mean()

        bw = torch.tensor(
            box_weights, dtype=torch.float32, device=s_seg_logits.device
        )
        return (torch.stack(box_losses) * bw).sum() / bw.sum().clamp_min(1e-6)

    @torch.no_grad()
    def _update_route_lambda(self, agree: float) -> float:
        """Open the trunk while the seg teacher stays near its best agreement
        with the frozen source anchor; close it once drift accumulates.

        Self-calibrating: compares the agreement EMA against its own running
        maximum, so no absolute threshold has to be guessed.
        """
        d = self.adaptive_routing_ema
        self._seg_agree_ema = (agree if self._seg_agree_ema is None
                               else d * self._seg_agree_ema + (1.0 - d) * agree)
        self._seg_agree_max = max(self._seg_agree_max, self._seg_agree_ema)
        ok = self._seg_agree_ema >= self.adaptive_routing_beta * self._seg_agree_max
        self._route_lambda = self.aux_trunk_lambda if ok else 0.0
        return self._route_lambda

    def _update_class_marginal(self, teacher_probs):
        """Running EMA of the teacher's predicted class marginal."""
        m = teacher_probs.mean(dim=(0, 2, 3)).float()
        if self._class_marginal is None:
            self._class_marginal = m.clone()
        else:
            a = self.class_marginal_ema
            self._class_marginal.mul_(a).add_(m, alpha=1.0 - a)
        return self._class_marginal

    # ------------------------------------------------------------------
    # Task-aware gradient combination on the shared trunk (S1/S2/S3/S5).
    # ------------------------------------------------------------------
    def _ensure_param_index(self):
        if self._param_index is None:
            self._param_index = [
                (n, p) for n, p in self.student.named_parameters() if p.requires_grad
            ]

            # Entire shared representation: ResNet bottom-up + FPN.
            self._shared_names = [
                n for n, _ in self._param_index
                if any(n.startswith(pfx) for pfx in _SHARED_PARAM_PREFIXES)
            ]
            self._shared_set = set(self._shared_names)

            # ResNet encoder only. In the new ``resnet_route`` mode auxiliary
            # gradients are scaled/blocked here, while the FPN stays fully MTL.
            self._resnet_names = [
                n for n, _ in self._param_index
                if any(n.startswith(pfx) for pfx in _RESNET_PARAM_PREFIXES)
            ]
            self._resnet_set = set(self._resnet_names)

            # Everything shared that is not bottom-up ResNet belongs to the FPN
            # (lateral/output convs, top block, etc.).
            self._fpn_names = [
                n for n in self._shared_names if n not in self._resnet_set
            ]
            self._fpn_set = set(self._fpn_names)

            # Preserve the old aux_head_only / E38 route-scope behaviour exactly.
            if self.route_scope == "fpn":
                self._routed_set = self._fpn_set
            else:
                self._routed_set = self._shared_set

            if self.conflict_mode == "resnet_route":
                print(
                    f"[CT-CMT-ROUTE] shared={len(self._shared_names)} "
                    f"resnet={len(self._resnet_names)} "
                    f"fpn={len(self._fpn_names)} "
                    f"lam={self._route_lambda:.2f}"
                )

    def _snapshot_grads(self):
        """Clone the current .grad of every trainable student param, then clear."""
        g = {n: p.grad.detach().clone() for n, p in self._param_index if p.grad is not None}
        self.optimizer.zero_grad(set_to_none=True)
        return g

    def _shared_dot(self, ga, gb) -> float:
        """<ga, gb> over shared-trunk parameters only, as one flattened vector."""
        acc = None
        for n in self._shared_names:
            a, b = ga.get(n), gb.get(n)
            if a is None or b is None:
                continue
            t = (a * b).sum()
            acc = t if acc is None else acc + t
        return 0.0 if acc is None else float(acc)

    def _subset_dot(self, ga, gb, names) -> float:
        """Dot product over an explicit parameter-name subset."""
        acc = None
        for n in names:
            a, b = ga.get(n), gb.get(n)
            if a is None or b is None:
                continue
            t = (a * b).sum()
            acc = t if acc is None else acc + t
        return 0.0 if acc is None else float(acc)

    def _shared_block_cos(self, ga, gb):
        """Per-block cos(ga, gb): where in the trunk the conflict actually lives."""
        acc = {}
        for n in self._shared_names:
            a, b = ga.get(n), gb.get(n)
            if a is None or b is None:
                continue
            d = acc.setdefault(_shared_block_of(n), [0.0, 0.0, 0.0])
            d[0] += float((a * b).sum())
            d[1] += float((a * a).sum())
            d[2] += float((b * b).sum())
        return {
            k: v[0] / (math.sqrt(max(v[1], 0.0) * max(v[2], 0.0)) + 1e-12)
            for k, v in acc.items()
        }

    @staticmethod
    def _task_coco_consensus_coeffs(
        g11: float,
        g12: float,
        g22: float,
        gc_c: float = 0.5,
        iters: int = 100,
        lr_default: float = 1.0,
        eps: float = 1e-4,
    ):
        """Return official CoCo Gradient-Consensus coefficients for two tasks.

        This is a memory-efficient port of leafheavy/MT-TTA's official
        ``GradientConsensus.gradient_consensus`` implementation.  The official
        code first flattens every task gradient and builds ``GG = grads @ grads.T``.
        Here the caller already provides that exact 2x2 Gram matrix through
        ``g11=<g_det,g_det>``, ``g12=<g_det,g_seg>``, ``g22=<g_seg,g_seg>``.

        The optimization itself intentionally mirrors the official repository:
          * mean gradient g0 through row/overall means of the normalized Gram;
          * softmax task weights optimized with Adam for 100 iterations by default;
          * objective ``<g_w,g_0> + phi * ||g_w||``;
          * final *unscaled* direction used by the released code.

        Returns
        -------
        c_det, c_seg, w_det, w_seg
            ``g_GC = c_det * g_det + c_seg * g_seg`` on shared parameters,
            while ``w_*`` are the simplex weights optimized by CoCo.
        """
        # The official implementation moves the tiny task Gram matrix to CPU
        # for the weight search.  Keeping this 2x2 optimization on CPU also
        # avoids retaining/flattening a second copy of the full backbone grads.
        GG = torch.tensor(
            [[g11, g12], [g12, g22]],
            dtype=torch.float32,
            device="cpu",
        )
        scale = (torch.diag(GG) + eps).sqrt().mean()
        GG = GG / scale.pow(2)

        Gg = GG.mean(1, keepdim=True)
        gg = Gg.mean(0, keepdim=True)

        w = torch.zeros(2, 1, requires_grad=True, device="cpu")
        w_opt = torch.optim.Adam(
            [w], lr=lr_default, betas=(0.9, 0.999), weight_decay=1e-4
        )
        phi_sqrt = (gg + eps).sqrt() * gc_c

        w_best = None
        obj_best = float("inf")
        for i in range(iters):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            term1 = ww.t().mm(Gg)
            term2 = (ww.t().mm(GG).mm(ww) + eps).sqrt()
            obj = term1 + phi_sqrt * term2

            obj_scalar = float(obj.item())
            if obj_scalar < obj_best:
                obj_best = obj_scalar
                w_best = w.detach().clone()
            if i < iters - 1:
                obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww) + eps).sqrt()
        lambda_frac = phi_sqrt.view(-1) / (gw_norm + eps)

        # This matches the released CoCo code exactly: the optional
        # /(1 + GC_c**2) scaling is commented out in GradientConsensus.py.
        coeffs = 0.5 + ww.view(-1) * lambda_frac.reshape(())
        return (
            float(coeffs[0].item()),
            float(coeffs[1].item()),
            float(ww[0].item()),
            float(ww[1].item()),
        )

    def _backward_task_coco(self, loss_dict):
        """Task-level CoCo consensus for detection vs semantic segmentation.

        The existing CT-CMT/CT-CR formulation is left untouched.  Only the
        *task* gradients on the shared Panoptic-FPN backbone are reconciled:

            g_det = d L_det / d theta_shared
            g_seg = d L_seg / d theta_shared
            g_shared = CoCo(g_det, g_seg) + g_regularizers

        where ``g_regularizers`` contains CT-CR, CT-CL, prototype loss, etc.
        Those cross-task/auxiliary terms keep their native full gradients and
        are deliberately NOT folded into ``g_seg``.  Task-specific heads also
        keep their native gradients; CoCo is applied only to ``backbone.*``
        (ResNet bottom-up + FPN), the parameters shared by both tasks.

        If either task loss is absent on a gated/empty-pseudo step, there is no
        two-task consensus problem and we fall back to the unchanged joint
        backward for that step.
        """
        diag = {}
        groups = {}
        for k, v in loss_dict.items():
            group = k.split("/")[0]
            groups[group] = v if group not in groups else groups[group] + v

        det_loss = groups.get("det")
        seg_loss = groups.get("seg")
        reg_keys = sorted(k for k in groups if k not in ("det", "seg"))

        # Faithful task-level consensus requires both task objectives.
        if (
            det_loss is None
            or seg_loss is None
            or not det_loss.requires_grad
            or not seg_loss.requires_grad
        ):
            total = sum(loss_dict.values())
            if total.requires_grad:
                total.backward()
            diag["task_coco_active"] = False
            return diag

        self._ensure_param_index()

        passes = [("det", det_loss), ("seg", seg_loss)]
        if reg_keys:
            reg_loss = sum(groups[k] for k in reg_keys)
            if reg_loss.requires_grad:
                passes.append(("reg", reg_loss))

        grads = {}
        for i, (name, loss) in enumerate(passes):
            loss.backward(retain_graph=(i < len(passes) - 1))
            grads[name] = self._snapshot_grads()

        g_det = grads["det"]
        g_seg = grads["seg"]
        g_reg = grads.get("reg", {})

        gdd = self._shared_dot(g_det, g_det)
        gds = self._shared_dot(g_det, g_seg)
        gss = self._shared_dot(g_seg, g_seg)
        nd = math.sqrt(max(gdd, 0.0))
        ns = math.sqrt(max(gss, 0.0))
        cos = gds / (nd * ns + 1e-12)
        conflict = gds < 0.0

        # Degenerate shared gradient: preserve the native update instead of
        # feeding an ill-defined task into the consensus search.
        if gdd <= 1e-24 or gss <= 1e-24:
            c_det = c_seg = 1.0
            w_det = w_seg = 0.5
            consensus_applied = False
        else:
            c_det, c_seg, w_det, w_seg = self._task_coco_consensus_coeffs(
                gdd,
                gds,
                gss,
                gc_c=self.task_coco_gc_c,
                iters=self.task_coco_iters,
                lr_default=self.task_coco_lr,
                eps=self.task_coco_eps,
            )
            consensus_applied = True

        nr = math.sqrt(max(self._shared_dot(g_reg, g_reg), 0.0)) if g_reg else 0.0
        diag.update({
            "task_coco_active": consensus_applied,
            "cos": cos,
            "conflict": conflict,
            "g_det": nd,
            "g_seg": ns,
            "g_reg": nr,
            "w_det": w_det,
            "w_seg": w_seg,
            "c_det": c_det,
            "c_seg": c_seg,
            "gc_c": self.task_coco_gc_c,
        })

        # CoCo changes only the two native task gradients on the shared
        # representation.  Cross-task regularizers (especially CT-CR) stay
        # exactly as they were: coefficient 1.0 everywhere they naturally flow.
        for n, p in self._param_index:
            gd = g_det.get(n)
            gs = g_seg.get(n)
            gr = g_reg.get(n)

            if gd is None and gs is None and gr is None:
                p.grad = None
                continue

            out = torch.zeros_like(p)
            if n in self._shared_set:
                if gd is not None:
                    out.add_(gd, alpha=c_det)
                if gs is not None:
                    out.add_(gs, alpha=c_seg)
            else:
                # Task-specific heads: native task gradients, no CoCo surgery.
                if gd is not None:
                    out.add_(gd)
                if gs is not None:
                    out.add_(gs)

            if gr is not None:
                out.add_(gr)
            p.grad = out

        s = self._conflict_stats
        s["steps"] += 1
        s["both"] += 1
        s["conflicts"] += int(conflict)
        s["projected"] += int(consensus_applied)
        s["cos_sum"] += cos
        s["w_det_sum"] += w_det
        s["w_seg_sum"] += w_seg
        s["c_det_sum"] += c_det
        s["c_seg_sum"] += c_seg
        return diag

    def _backward_and_combine(self, loss_dict):
        """Backward pass(es) + shared-trunk gradient combination.

        On return the student's ``.grad`` fields hold exactly what the optimizer
        should apply. Returns a diagnostics dict (logging only).
        """
        if self.conflict_mode == "task_coco":
            return self._backward_task_coco(loss_dict)
        if self.fisher_xtask:
            return self._backward_fisher_xtask(loss_dict)

        diag = {}
        groups = {}
        for k, v in loss_dict.items():
            g = k.split("/")[0]
            groups[g] = v if g not in groups else groups[g] + v
        det_loss = groups.get("det")
        aux_keys = sorted(k for k in groups if k != "det")

        want_components = self.grad_diag and (self.iter % self.grad_diag_every == 0)
        if self.conflict_mode == "none" and not want_components:
            total = sum(loss_dict.values())
            if total.requires_grad:
                total.backward()
            return diag
        if det_loss is None or not aux_keys:
            # Only one side is present this step: nothing to reconcile.
            total = sum(loss_dict.values())
            if total.requires_grad:
                total.backward()
                if self.conflict_mode == "aux_head_only" and det_loss is None:
                    # Preserve the legacy S6/E38 path exactly.
                    self._ensure_param_index()
                    lam = self._route_lambda
                    for n, p in self._param_index:
                        if n not in self._routed_set or p.grad is None:
                            continue
                        p.grad = None if lam == 0.0 else p.grad * lam
                    self._conflict_stats["aux_only_routed"] = (
                        self._conflict_stats.get("aux_only_routed", 0) + 1)

                elif self.conflict_mode == "resnet_route" and det_loss is None:
                    # No detector gradient on this step: still prevent the
                    # segmentation/CT-CR objective from bypassing the route and
                    # updating ResNet at full strength. FPN + aux heads stay open.
                    self._ensure_param_index()
                    lam = self._route_lambda
                    for n, p in self._param_index:
                        if n not in self._resnet_set or p.grad is None:
                            continue
                        p.grad = torch.zeros_like(p.grad) if lam == 0.0 else p.grad * lam
                    self._conflict_stats["aux_only_resnet_routed"] = (
                        self._conflict_stats.get("aux_only_resnet_routed", 0) + 1)
            return diag

        self._ensure_param_index()
        if want_components:
            passes = [("det", det_loss)] + [(k, groups[k]) for k in aux_keys]
        else:
            passes = [("det", det_loss), ("aux", sum(groups[k] for k in aux_keys))]

        # A degenerate CT-CL (no positive pairs) returns a detached constant.
        # It contributes no gradient, but backwarding it on its own raises.
        passes = [(n, l) for n, l in passes if l.requires_grad]
        names = {n for n, _ in passes}
        if "det" not in names or len(names) < 2:
            if passes:
                sum(l for _, l in passes).backward()
            return diag

        grads = {}
        for i, (name, loss) in enumerate(passes):
            loss.backward(retain_graph=(i < len(passes) - 1))
            grads[name] = self._snapshot_grads()

        g_det = grads.pop("det")
        g_aux = {}
        for comp in grads.values():
            for n, t in comp.items():
                g_aux[n] = t.clone() if n not in g_aux else g_aux[n] + t

        nd2 = self._shared_dot(g_det, g_det)
        nd = math.sqrt(max(nd2, 0.0))
        if want_components:
            # Norms are over shared-trunk params only: each component's share of
            # the gradient that can carry negative transfer.
            diag["gnorm_det"] = nd
            for name, comp in grads.items():
                nc = math.sqrt(max(self._shared_dot(comp, comp), 0.0))
                diag[f"gnorm_{name}"] = nc
                diag[f"cos_det_{name}"] = self._shared_dot(g_det, comp) / (nd * nc + 1e-12)

        dot = self._shared_dot(g_det, g_aux)
        na2 = self._shared_dot(g_aux, g_aux)
        na = math.sqrt(max(na2, 0.0))
        cos = dot / (nd * na + 1e-12)
        conflict = dot < 0.0
        diag.update({"cos": cos, "g_det": nd, "g_aux": na, "conflict": conflict})

        # Route-specific diagnostics. These are pre-routing norms; the applied
        # auxiliary ResNet norm is exactly lambda * pre-routing norm, while FPN
        # remains unchanged in ``resnet_route``.
        if self.conflict_mode == "resnet_route" or want_components:
            det_res2 = self._subset_dot(g_det, g_det, self._resnet_names)
            aux_res2 = self._subset_dot(g_aux, g_aux, self._resnet_names)
            det_fpn2 = self._subset_dot(g_det, g_det, self._fpn_names)
            aux_fpn2 = self._subset_dot(g_aux, g_aux, self._fpn_names)
            diag["g_det_resnet"] = math.sqrt(max(det_res2, 0.0))
            diag["g_aux_resnet"] = math.sqrt(max(aux_res2, 0.0))
            diag["g_det_fpn"] = math.sqrt(max(det_fpn2, 0.0))
            diag["g_aux_fpn"] = math.sqrt(max(aux_fpn2, 0.0))

        if want_components:
            diag["blocks"] = self._shared_block_cos(g_det, g_aux)

        # Per-mode scalar coefficients: shared grad = c_det*g_det + c_aux*g_aux.
        c_det, c_aux_shared, c_aux_other = 1.0, 1.0, 1.0
        if self.conflict_mode == "protect_det":
            if conflict:
                coef = dot / (nd2 + 1e-12)
                for n in self._shared_names:
                    a, d = g_aux.get(n), g_det.get(n)
                    if a is not None and d is not None:
                        a.add_(d, alpha=-coef)
                diag["projected"] = True
                diag["g_aux_proj"] = math.sqrt(max(self._shared_dot(g_aux, g_aux), 0.0))
            else:
                diag["projected"] = False
                diag["g_aux_proj"] = na
        elif self.conflict_mode == "cagrad":
            x = _cagrad_weight(nd2, dot, na2, self.cagrad_alpha)
            gw2 = x * x * nd2 + 2.0 * x * (1.0 - x) * dot + (1.0 - x) * (1.0 - x) * na2
            g0 = math.sqrt(max(0.25 * (nd2 + 2.0 * dot + na2), 0.0))
            coef = self.cagrad_alpha * g0 / (math.sqrt(max(gw2, 0.0)) + 1e-8)
            # Official CAGrad averages the task gradients; the x2 restores the
            # magnitude of the joint (summed) baseline so step size is unchanged.
            denom = 1.0 + self.cagrad_alpha ** 2
            c_det = (1.0 + 2.0 * coef * x) / denom
            c_aux_shared = (1.0 + 2.0 * coef * (1.0 - x)) / denom
            diag.update({"w_det": x, "w_aux": 1.0 - x,
                         "c_det": c_det, "c_aux": c_aux_shared})
        elif self.conflict_mode == "hard_decouple":
            c_aux_shared = 0.0 if conflict else 1.0
            diag["decoupled"] = conflict
        elif self.conflict_mode == "aux_head_only":
            # Detection owns the selected shared representation; the auxiliary
            # objective reaches it only through lambda (legacy S6/E38 path).
            c_aux_shared = self._route_lambda
            diag["decoupled"] = True
            diag["lam"] = c_aux_shared
        elif self.conflict_mode == "resnet_route":
            # New ResNet-block / FPN-open path:
            #   ResNet: g_det + lambda * g_aux
            #   FPN:    g_det + g_aux
            #   heads:  their native full gradients
            # lambda=0 is the static detector-owned ResNet experiment.
            diag["resnet_routed"] = True
            diag["lam"] = self._route_lambda
            diag["g_aux_resnet_applied"] = (
                self._route_lambda * diag.get("g_aux_resnet", 0.0)
            )
            diag["g_aux_fpn_applied"] = diag.get("g_aux_fpn", 0.0)
        elif self.conflict_mode == "dyn_weight":
            gamma = max(0.0, cos)
            c_aux_shared = c_aux_other = gamma
            diag["gamma"] = gamma

        for n, p in self._param_index:
            if self.conflict_mode == "resnet_route":
                # Detection keeps full authority everywhere. Auxiliary gradients
                # are routed only through the ResNet bottom-up; FPN and task
                # heads retain the full auxiliary signal.
                cd = 1.0
                ca = self._route_lambda if n in self._resnet_set else 1.0
            else:
                routed = n in self._routed_set
                cd = c_det if routed else 1.0
                ca = c_aux_shared if routed else c_aux_other

            gd, ga = g_det.get(n), g_aux.get(n)
            if gd is None and ga is None:
                p.grad = None
                continue
            # A zeroed coefficient must still yield a zero grad, not None, or
            # SGD momentum would silently skip the parameter instead of decaying.
            out = torch.zeros_like(p)
            if gd is not None and cd != 0.0:
                out.add_(gd, alpha=cd)
            if ga is not None and ca != 0.0:
                out.add_(ga, alpha=ca)
            p.grad = out

        s = self._conflict_stats
        s["steps"] += 1
        s["both"] += 1
        s["conflicts"] += int(conflict)
        s["projected"] += int(
            diag.get("projected", False)
            or diag.get("decoupled", False)
            or diag.get("resnet_routed", False)
        )
        s["cos_sum"] += cos
        s["gamma_sum"] += diag.get("gamma", 0.0)
        s["w_det_sum"] += diag.get("w_det", 0.0)
        return diag

    # ------------------------------------------------------------------
    # Experimental: simple Fisher cross-task routing (DET vs AUX).
    # AUX = seg + anchor + CT-CR (every non-"det" loss group).
    # ------------------------------------------------------------------
    _FX_BLOCKS = ("res3", "res4", "res5")
    # stats columns: n, sumF_det, sumF_aux, det_gt, aux_gt, conflict,
    #                det_prot, aux_prot, sum_c_det, sum_c_aux, n_routed
    _FX_NCOL = 11

    def _backward_fisher_xtask(self, loss_dict):
        """Two backwards (DET, AUX); ResNet res3-5 routed by lagged Fisher.

        route_t = f(F_{t-1}, g_det_t, g_aux_t); only afterwards are the EMAs
        updated with the UNMODIFIED current task gradients. Steps without a
        usable DET or AUX gradient use the plain joint gradient and update
        nothing. FPN / heads always receive g_det + g_aux.
        """
        groups = {}
        for k, v in loss_dict.items():
            g = k.split("/")[0]
            groups[g] = v if g not in groups else groups[g] + v
        det_loss = groups.get("det")
        aux_keys = sorted(k for k in groups if k != "det")
        aux_loss = sum(groups[k] for k in aux_keys) if aux_keys else None
        if (det_loss is None or aux_loss is None
                or not det_loss.requires_grad or not aux_loss.requires_grad):
            total = sum(loss_dict.values())
            if total.requires_grad:
                total.backward()
            return {}

        self._ensure_param_index()
        det_loss.backward(retain_graph=True)
        g_det = self._snapshot_grads()
        aux_loss.backward()
        g_aux = self._snapshot_grads()

        if self._fx_F_det is None:
            self._fx_F_det = {n: torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                              for n, p in self._param_index if n in self._resnet_set}
            self._fx_F_aux = {n: torch.zeros_like(f) for n, f in self._fx_F_det.items()}
            self._fx_stats = torch.zeros(len(self._FX_BLOCKS), self._FX_NCOL,
                                         dtype=torch.float64,
                                         device=next(iter(self._fx_F_det.values())).device)
        routing_on = self._fx_n_active >= self.fx_warmup
        lam, beta, eps = self.fx_lambda, self.fx_beta, 1e-30

        for n, p in self._param_index:
            gd, ga = g_det.get(n), g_aux.get(n)
            if gd is None and ga is None:
                p.grad = None
                continue
            if n not in self._fx_F_det:
                out = torch.zeros_like(p)
                if gd is not None:
                    out.add_(gd)
                if ga is not None:
                    out.add_(ga)
                p.grad = out
                continue

            gd = torch.zeros_like(p) if gd is None else gd
            ga = torch.zeros_like(p) if ga is None else ga
            Fd, Fa = self._fx_F_det[n], self._fx_F_aux[n]   # F_{t-1}
            if routing_on:
                Fdn = Fd / (Fd.mean() + eps)
                Fan = Fa / (Fa.mean() + eps)
                conflict = (gd * ga) < 0
                near = (Fdn - Fan).abs() <= 1e-3 * torch.maximum(Fdn, Fan)
                det_prot = conflict & (Fdn > Fan) & ~near   # keep DET, scale AUX
                aux_prot = conflict & (Fan > Fdn) & ~near   # keep AUX, scale DET
                c_det = 1.0 - (1.0 - lam) * aux_prot.to(gd.dtype)
                c_aux = 1.0 - (1.0 - lam) * det_prot.to(ga.dtype)
                out = gd * c_det + ga * c_aux
                blk = _shared_block_of(n)
                if blk in self._FX_BLOCKS:
                    row = self._fx_stats[self._FX_BLOCKS.index(blk)]
                    row += torch.stack([
                        torch.tensor(float(p.numel()), device=row.device,
                                     dtype=torch.float64),
                        Fd.sum().double(), Fa.sum().double(),
                        (Fdn > Fan).sum().double(), (Fan > Fdn).sum().double(),
                        conflict.sum().double(), det_prot.sum().double(),
                        aux_prot.sum().double(), c_det.sum().double(),
                        c_aux.sum().double(),
                        torch.tensor(1.0, device=row.device, dtype=torch.float64),
                    ])
            else:
                out = gd + ga
            p.grad = out
            # Lagged update: the unmodified current gradients enter F only now.
            Fd.mul_(beta).addcmul_(gd.float(), gd.float(), value=1.0 - beta)
            Fa.mul_(beta).addcmul_(ga.float(), ga.float(), value=1.0 - beta)

        self._fx_n_active += 1
        if routing_on:
            self._fx_stat_steps += 1
        return {}

    def _fx_flush_log(self):
        """Print + JSONL the accumulated routing statistics, then reset."""
        if not self.fisher_xtask or self._fx_stats is None:
            return
        st = self._fx_stats.cpu().tolist()
        rec = {"iter": self.iter, "det_active_updates": self._fx_n_active,
               "warmup": self.fx_warmup, "routing_on": self._fx_n_active > self.fx_warmup,
               "routed_steps_in_window": self._fx_stat_steps, "blocks": {}}
        tot = [0.0] * self._FX_NCOL
        for b, row in zip(self._FX_BLOCKS, st):
            for i, v in enumerate(row):
                tot[i] += v
            rec["blocks"][b] = self._fx_summarize(row)
        rec["all"] = self._fx_summarize(tot)
        a = rec["all"]
        if a:
            print(f"[CT-CMT-FXTASK] iter={self.iter} det_active={self._fx_n_active} "
                  f"F_det={a['mean_F_det']:.3e} F_aux={a['mean_F_aux']:.3e} "
                  f"Fdet>Faux={a['frac_Fdet_gt_Faux']:.3f} Faux>Fdet={a['frac_Faux_gt_Fdet']:.3f} "
                  f"conflict={a['frac_conflict']:.3f} det_prot={a['frac_det_protect']:.4f} "
                  f"aux_prot={a['frac_aux_protect']:.4f} c_det={a['mean_c_det']:.4f} "
                  f"c_aux={a['mean_c_aux']:.4f} | "
                  + " ".join(f"{b}:conf={r['frac_conflict']:.3f}/dp={r['frac_det_protect']:.4f}"
                             f"/ap={r['frac_aux_protect']:.4f}"
                             for b, r in rec["blocks"].items() if r))
        elif self._fx_n_active > 0:
            print(f"[CT-CMT-FXTASK] iter={self.iter} warm-up det_active="
                  f"{self._fx_n_active}/{self.fx_warmup}")
        try:
            d = os.path.join(self._fx_out_dir, "fisher_xtask.jsonl")
            os.makedirs(self._fx_out_dir, exist_ok=True)
            with open(d, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:
            print(f"[CT-CMT-FXTASK] jsonl write failed: {exc}")
        self._fx_stats.zero_()
        self._fx_stat_steps = 0

    @staticmethod
    def _fx_summarize(row):
        n = row[0]
        if n <= 0:
            return {}
        return {"n_units": n, "mean_F_det": row[1] / n, "mean_F_aux": row[2] / n,
                "frac_Fdet_gt_Faux": row[3] / n, "frac_Faux_gt_Fdet": row[4] / n,
                "frac_conflict": row[5] / n, "frac_det_protect": row[6] / n,
                "frac_aux_protect": row[7] / n,
                "mean_c_det": row[8] / n, "mean_c_aux": row[9] / n}

    # ------------------------------------------------------------------
    # EMA + stochastic restore.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_teacher(self):
        d = self.ema_decay
        s_state = self.student.state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s_state[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s_state[k])

    @torch.no_grad()
    def _stochastic_restore(self, fisher=None):
        if self.restore_prob <= 0.0:
            return
        if self.fisher_guided_restore:
            return self._fisher_guided_restore()
        # Fisher mode needs the gradients from this step; skip when none exist.
        if self.fisher_restore and not fisher:
            return
        _shared_prefixes = ("backbone.", "fpn.", "proposal_generator.anchor_generator.")
        # E5: measure current shared-trunk drift and adapt the backbone-restore factor.
        eff_backbone_factor = self.backbone_rst_factor
        if self.adaptive_str and self.cross_task_fisher:
            if self._backbone_source_norm is None:
                self._backbone_source_norm = 0.0
                for key, src in self._source_params.items():
                    if any(key.startswith(pfx) for pfx in _shared_prefixes):
                        self._backbone_source_norm += float(src.pow(2).sum().cpu())
                self._backbone_source_norm = math.sqrt(self._backbone_source_norm) + 1e-6
            drift_sq = 0.0
            student_state = self.student.state_dict()
            for key, src in self._source_params.items():
                if any(key.startswith(pfx) for pfx in _shared_prefixes):
                    if key in student_state:
                        s = student_state[key]
                        drift_sq += float((s - src.to(s.device)).pow(2).sum().cpu())
            drift = math.sqrt(drift_sq) / self._backbone_source_norm
            # Sigmoid boost: base + boost * σ(10*(drift-pivot))
            eff_backbone_factor = self.adaptive_str_base + self.adaptive_str_boost * (
                1.0 / (1.0 + math.exp(-10.0 * (drift - self.adaptive_str_pivot)))
            )
        for nm, m in self.student.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name not in ("weight", "bias") or not p.requires_grad:
                    continue
                key = f"{nm}.{np_name}"
                src = self._source_params.get(key)
                if src is None:
                    continue
                rst = self.restore_prob
                if self.cross_task_fisher and any(key.startswith(pfx) for pfx in _shared_prefixes):
                    rst = rst * eff_backbone_factor
                if rst <= 0.0:
                    continue
                if self.fisher_restore:
                    g = fisher.get(key)
                    if g is None:
                        continue
                    mask = _fisher_restore_mask(g, rst)
                else:
                    mask = (torch.rand_like(p) < rst).float()
                src_dev = src.to(p.device, non_blocking=True)
                p.data.mul_(1.0 - mask).add_(src_dev * mask)

    @torch.no_grad()
    def _fisher_guided_restore(self):
        """Random-budget-matched restore that prefers LOW-Fisher entries.

        Per trainable weight/bias tensor (same set and same per-tensor
        probability ``rst`` as the random baseline) draw k ~ Binomial(n, rst)
        and restore the k entries with the smallest (F/Fmax + 1e-8) * U, where
        F = (final combined gradient)^2 and U ~ Uniform(0,1). Tensors without a
        gradient (or steps without any loss) have F = 0, i.e. plain random.
        Gradients, losses, LR and the optimizer are never touched.
        """
        shared_prefixes = ("backbone.", "fpn.", "proposal_generator.anchor_generator.")
        dev_stats = None
        for nm, m in self.student.named_modules():
            for np_name, p in m.named_parameters(recurse=False):
                if np_name not in ("weight", "bias") or not p.requires_grad:
                    continue
                key = f"{nm}.{np_name}"
                src = self._source_params.get(key)
                if src is None:
                    continue
                rst = self.restore_prob
                if self.cross_task_fisher and any(key.startswith(x) for x in shared_prefixes):
                    rst = rst * self.backbone_rst_factor
                if rst <= 0.0:
                    continue
                n = p.numel()
                # CPU binomial: no GPU sync; same count law as Bernoulli(rst) mask.
                k = int(torch.binomial(torch.tensor([float(n)]),
                                       torch.tensor([float(rst)])).item())
                if k <= 0:
                    continue
                pf = p.data.view(-1)
                sf = src.to(p.device, non_blocking=True).view(-1)
                if k >= n:
                    pf.copy_(sf)
                    continue
                if self._fr_has_grad and p.grad is not None:
                    F = p.grad.detach().float().pow(2).view(-1)
                    Fn = F / F.max().clamp_min(1e-30)
                else:
                    Fn = torch.zeros(n, device=p.device)
                score = (Fn + 1e-8) * torch.rand_like(Fn)
                idx = torch.topk(score, k, largest=False).indices
                pf[idx] = sf[idx]
                # sanity stat: mean normalised F of restored vs all entries
                st = torch.stack([Fn[idx].sum().double(), Fn.mean().double() * k,
                                  torch.tensor(float(k), device=p.device,
                                               dtype=torch.float64)])
                dev_stats = st if dev_stats is None else dev_stats + st
        if dev_stats is not None:
            self._fr_stats = dev_stats if self._fr_stats is None else self._fr_stats + dev_stats

    def _fr_flush_log(self):
        if not self.fisher_guided_restore or self._fr_stats is None:
            return
        sel, exp, k = self._fr_stats.cpu().tolist()
        print(f"[CT-CMT-FRESTORE] iter={self.iter} restored={int(k)} "
              f"selF/randF={(sel / exp) if exp > 0 else float('nan'):.4f} "
              f"(<1 => restored entries have lower Fisher than random)")
        self._fr_stats.zero_()

    def _diag_write(self, record):
        """Append one diagnostic record as JSONL.

        Diagnostics are intentionally best-effort and must never change
        the adaptation trajectory.
        """
        if not self.diag_jsonl:
            return

        try:
            diag_dir = os.path.dirname(self.diag_jsonl)
            if diag_dir:
                os.makedirs(diag_dir, exist_ok=True)

            with open(self.diag_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as exc:
            print(f"[CT-CMT-DIAG] failed to write diagnostics: {exc}")

    @staticmethod
    def _diag_cityscapes_key(file_name):
        name = os.path.basename(str(file_name or ""))
        suffix = "_leftImg8bit.png"
        if name.endswith(suffix):
            return name[:-len(suffix)]
        return os.path.splitext(name)[0]

    @torch.no_grad()
    def _diag_masked_ctcr_stats(self, instances, teacher_seg_probs, file_name):
        """Diagnostic-only CT-CR spatial-support statistics against semantic GT.

        For every pre-CTPV pseudo-box and every configured probability threshold,
        log:
          - box_pixels: number of pixels inside the box on the seg grid
          - gt_pixels: GT pixels of the mapped semantic class inside the box
          - sweep: SEGMENTATION-derived support, teacher P(class) >= tau
          - mask_sweep: DETECTION-derived support, teacher mask head >= tau
        each entry carrying mask_pixels and its intersection with the GT pixels.

        The rectangle baseline (CT-CR mode A) needs no sweep: it covers the whole
        box, so mask_pixels == box_pixels and intersection == gt_pixels.

        These values are used only for post-hoc analysis.
        """
        if (
            not self.mask_diag_gt_root
            or not self._mask_diag_gt_index
            or teacher_seg_probs is None
            or len(instances[0]) == 0
        ):
            return []

        key = self._diag_cityscapes_key(file_name)
        gt_path = self._mask_diag_gt_index.get(key)
        if gt_path is None:
            return []

        try:
            gt_np = np.asarray(Image.open(gt_path).convert("L"), dtype=np.int64)
        except Exception as exc:
            print(f"[CT-CMT-DIAG] failed to load semantic GT {gt_path}: {exc}")
            return []

        inst = instances[0]
        _, K, H, W = teacher_seg_probs.shape

        # Detector-derived spatial prior. Present whenever the teacher keeps its
        # mask head; (N, 1, M, M) posteriors in the ROI frame, not yet pasted.
        pred_masks = inst.pred_masks if inst.has("pred_masks") else None

        # Resize semantic trainIds to exactly the teacher-probability grid.
        gt_t = torch.from_numpy(gt_np).to(teacher_seg_probs.device)
        gt_t = gt_t[None, None].float()
        gt_t = F.interpolate(gt_t, size=(H, W), mode="nearest")[0, 0].long()

        img_h, img_w = inst.image_size
        sx = W / max(img_w, 1)
        sy = H / max(img_h, 1)

        out = []

        for j, (box, c) in enumerate(
            zip(
                inst.pred_boxes.tensor.tolist(),
                inst.pred_classes.tolist(),
            )
        ):
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                continue

            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K:
                continue

            x1, y1, x2, y2 = box
            x1i = max(int(round(x1 * sx)), 0)
            y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W)
            y2i = min(int(round(y2 * sy)), H)

            if x2i <= x1i or y2i <= y1i:
                continue

            q = teacher_seg_probs[0, seg_c, y1i:y2i, x1i:x2i]
            gt_crop = gt_t[y1i:y2i, x1i:x2i]
            gt_mask = gt_crop == int(seg_c)

            box_pixels = int(q.numel())
            gt_pixels = int(gt_mask.sum().item())

            sweep = []
            for tau in self.mask_diag_thresholds:
                pred_mask = q >= float(tau)
                mask_pixels = int(pred_mask.sum().item())
                intersection = int((pred_mask & gt_mask).sum().item())
                sweep.append({
                    "tau": float(tau),
                    "mask_pixels": mask_pixels,
                    "intersection": intersection,
                })

            mask_sweep = []
            mean_mask_prob = None
            if pred_masks is not None:
                m = F.interpolate(
                    pred_masks[j:j + 1].float(),
                    size=(y2i - y1i, x2i - x1i),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                mean_mask_prob = float(m.mean().item())
                for tau in self.mask_diag_thresholds:
                    pm = m >= float(tau)
                    mask_sweep.append({
                        "tau": float(tau),
                        "mask_pixels": int(pm.sum().item()),
                        "intersection": int((pm & gt_mask).sum().item()),
                    })

            out.append({
                "box_idx": int(j),
                "det_class": int(c),
                "seg_class": int(seg_c),
                "det_score": float(inst.scores[j].item()),
                "box_pixels": box_pixels,
                "gt_pixels": gt_pixels,
                "mean_class_prob": float(q.mean().item()),
                "mean_mask_prob": mean_mask_prob,
                "sweep": sweep,
                "mask_sweep": mask_sweep,
            })

        return out

    # V3: reject a pseudo-box if the seg head disagrees with its class.
    @torch.no_grad()
    def _ctpv_filter(self, instances, teacher_seg_probs, return_stats=False):
        """Cross-task pseudo-label verification.

        Keep a detection pseudo-box only when the segmentation teacher assigns
        at least self.ctpv_thresh fraction of the pixels inside the box to the
        corresponding semantic class.

        When return_stats=True, also return per-box diagnostic information.
        The filtering decision itself is unchanged.
        """
        inst = instances[0]

        # ------------------------------------------------------------
        # Nothing to verify.
        # ------------------------------------------------------------
        if len(inst) == 0 or teacher_seg_probs is None:
            if return_stats:
                return instances, []
            return instances

        _, K, H, W = teacher_seg_probs.shape

        img_h, img_w = inst.image_size
        sx = W / max(img_w, 1)
        sy = H / max(img_h, 1)

        keep = []
        records = []

        # ------------------------------------------------------------
        # Evaluate every pseudo-box independently.
        # ------------------------------------------------------------
        for j, (box, c) in enumerate(
            zip(
                inst.pred_boxes.tensor.tolist(),
                inst.pred_classes.tolist(),
            )
        ):
            det_score = float(inst.scores[j].item())

            # Detection class has no semantic mapping.
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                keep.append(True)

                if return_stats:
                    records.append({
                        "box_idx": int(j),
                        "det_class": int(c),
                        "det_score": det_score,
                        "box_xyxy": [float(v) for v in box],
                        "seg_class": None,
                        "agreement": None,
                        "keep": True,
                        "reason": "unmapped_detection_class",
                    })

                continue

            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]

            # Semantic class index not available in this segmentation head.
            if seg_c >= K:
                keep.append(True)

                if return_stats:
                    records.append({
                        "box_idx": int(j),
                        "det_class": int(c),
                        "det_score": det_score,
                        "box_xyxy": [float(v) for v in box],
                        "seg_class": int(seg_c),
                        "agreement": None,
                        "keep": True,
                        "reason": "seg_class_out_of_range",
                    })

                continue

            # --------------------------------------------------------
            # Map box from resized-image coordinates to seg-logit grid.
            # --------------------------------------------------------
            x1, y1, x2, y2 = box

            x1i = max(int(round(x1 * sx)), 0)
            y1i = max(int(round(y1 * sy)), 0)
            x2i = min(int(round(x2 * sx)), W)
            y2i = min(int(round(y2 * sy)), H)

            # Invalid / empty box.
            if x2i <= x1i or y2i <= y1i:
                keep.append(False)

                if return_stats:
                    records.append({
                        "box_idx": int(j),
                        "det_class": int(c),
                        "det_score": det_score,
                        "box_xyxy": [float(v) for v in box],
                        "seg_class": int(seg_c),
                        "agreement": 0.0,
                        "keep": False,
                        "reason": "invalid_box",
                    })

                continue

            # --------------------------------------------------------
            # Current CTPV definition:
            #
            # fraction of pixels inside the box whose semantic argmax
            # agrees with the detector class.
            # --------------------------------------------------------
            region = teacher_seg_probs[
                0,
                :,
                y1i:y2i,
                x1i:x2i,
            ]  # (K, h, w)

            pred_class = region.argmax(dim=0)

            agreement = float(
                (pred_class == seg_c)
                .float()
                .mean()
                .item()
            )

            keep_decision = agreement >= self.ctpv_thresh
            keep.append(keep_decision)

            if return_stats:
                records.append({
                    "box_idx": int(j),
                    "det_class": int(c),
                    "det_score": det_score,
                    "box_xyxy": [float(v) for v in box],
                    "seg_class": int(seg_c),
                    "agreement": agreement,
                    "keep": bool(keep_decision),
                    "reason": "agreement_threshold",
                })

        # ------------------------------------------------------------
        # Apply exactly the same binary CTPV filtering as before.
        # ------------------------------------------------------------
        if all(keep):
            result = instances

        else:
            keep_t = torch.tensor(
                keep,
                device=inst.pred_boxes.tensor.device,
                dtype=torch.bool,
            )

            filtered = Instances(inst.image_size)
            filtered.pred_boxes = Boxes(inst.pred_boxes.tensor[keep_t])
            filtered.pred_classes = inst.pred_classes[keep_t]
            filtered.scores = inst.scores[keep_t]

            result = [filtered]

        # ------------------------------------------------------------
        # Diagnostics are optional.
        # ------------------------------------------------------------
        if return_stats:
            return result, records

        return result

    # V4: update cross-task prototype anchors; returns prototype pull loss.
    def _proto_anchor_update_and_loss(self, features, pseudo_inst, teacher_seg_probs):
        inst = pseudo_inst[0]
        if len(inst) == 0:
            return features[next(iter(features))].new_zeros(())
        feat_key = self.student.roi_heads.box_in_features[-1]
        feat = features[feat_key]  # (B, C, H, W)
        stride = self.student.backbone.output_shape()[feat_key].stride
        boxes = (inst.pred_boxes.tensor / float(stride)).detach()
        classes = inst.pred_classes.detach().long().tolist()
        if teacher_seg_probs is not None:
            probs_feat = F.interpolate(teacher_seg_probs, size=feat.shape[-2:],
                                        mode="bilinear", align_corners=False)
        else:
            probs_feat = None
        H, W = feat.shape[-2:]
        proto_loss = feat.new_zeros(())
        for b, c in zip(boxes.tolist(), classes):
            if not (0 <= c < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                continue
            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            x1, y1, x2, y2 = [max(int(round(v)), 0) for v in b]
            x2 = min(x2, W); y2 = min(y2, H)
            if x2 <= x1 or y2 <= y1:
                continue
            # Det view.
            z_det = feat[0:1, :, y1:y2, x1:x2].mean(dim=(2, 3)).squeeze(0)
            z_det_n = F.normalize(z_det, dim=0)
            # Seg view (mask-weighted).
            if probs_feat is not None and seg_c < probs_feat.shape[1]:
                w = probs_feat[0, seg_c, y1:y2, x1:x2].clamp(min=1e-6)
                w = w / w.sum()
                z_seg = (feat[0, :, y1:y2, x1:x2] * w.unsqueeze(0)).sum(dim=(1, 2))
            else:
                z_seg = z_det.clone()
            z_seg_n = F.normalize(z_seg, dim=0)
            # Only update prototype if both views roughly agree (cosine sim > 0).
            cross_sim = float((z_det_n * z_seg_n).sum().item())
            with torch.no_grad():
                if cross_sim > 0.0:
                    z_det_np = z_det_n.detach()
                    z_seg_np = z_seg_n.detach()
                    if c not in self._det_protos:
                        self._det_protos[c] = z_det_np.clone()
                        self._seg_protos[c] = z_seg_np.clone()
                    else:
                        a = self.proto_ema
                        self._det_protos[c] = a * self._det_protos[c] + (1 - a) * z_det_np
                        self._seg_protos[c] = a * self._seg_protos[c] + (1 - a) * z_seg_np
            # Pull current features toward stored prototypes.
            if c in self._det_protos:
                proto_det = self._det_protos[c].to(feat.device)
                proto_seg = self._seg_protos[c].to(feat.device)
                proto_loss = proto_loss + (1.0 - (z_det_n * proto_det.detach()).sum())
                proto_loss = proto_loss + (1.0 - (z_seg_n * proto_seg.detach()).sum())
        return proto_loss

    # ------------------------------------------------------------------
    # Forward = one CTTA step. Returns per-image dict with 'sem_seg' and
    # 'instances' (no panoptic combine — avoids pred_masks dependency).
    # ------------------------------------------------------------------
    # CoTTA-style multi-scale aug-averaged teacher seg probs.
    # Only invoked when seg_aug_enabled AND anchor confidence is below the
    # per-image threshold, keeping runtime low on easy inputs.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _teacher_aug_avg_seg_probs(self, image_tensor):
        H, W = image_tensor.shape[-2:]
        divisor = int(getattr(self.teacher.backbone, "size_divisibility", 32) or 32)
        def _snap(v):
            v = max(int(round(v)), divisor)
            return ((v + divisor - 1) // divisor) * divisor
        accum = None
        n = 0
        for s in self.seg_aug_scales:
            Hs, Ws = _snap(H * s), _snap(W * s)
            for flip in self.seg_aug_flips:
                x_aug = F.interpolate(image_tensor, size=(Hs, Ws),
                                      mode="bilinear", align_corners=False)
                if flip:
                    x_aug = torch.flip(x_aug, dims=[-1])
                feats = self.teacher.backbone(x_aug)
                logits, _ = self.teacher.sem_seg_head(feats, None)
                logits = F.interpolate(logits, size=(Hs, Ws),
                                       mode="bilinear", align_corners=False)
                probs = logits.float().softmax(dim=1)
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                probs = F.interpolate(probs, size=(H, W),
                                      mode="bilinear", align_corners=False)
                accum = probs if accum is None else accum + probs
                n += 1
        return accum / float(max(n, 1))

    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward(self, batched_inputs):
        self.iter += 1

        # Trainer.test() puts every submodule into .eval() before running
        # inference_on_dataset; proposal_generator / roi_heads only emit losses
        # when .training is True, so put the student back in train mode here.
        # sem_seg_head is kept in eval() because we consume raw logits.
        self.student.train()
        if not self.det_only and hasattr(self.student, "sem_seg_head"):
            self.student.sem_seg_head.eval()

        # Predict-then-adapt protocol: freeze the weak-view teacher prediction
        # before any update of this frame.
        pre_update_out = None
        if self.predict_before_update:
            assert hasattr(self.teacher, "roi_heads"), \
                "CTCMT_PREDICT_BEFORE_UPDATE needs a detector-bearing teacher"
            with torch.no_grad():
                pre_update_out = self.teacher.inference(batched_inputs, do_postprocess=False)

        # 1. Teacher pseudo-labels.
        pseudo_inst, teacher_sem_results, keep_step = self._teacher_pseudo(batched_inputs)

        # V1: per-task confidence gates. Compute teacher seg confidence
        # independently so each task branch can fire/skip on its own.
        det_gate = keep_step  # default: same as global gate
        seg_gate = not self.det_only  # default: always run unless det_only
        if self.per_task_gate and not self.det_only and not self.seg_only:
            with torch.no_grad():
                # Det gate: skip if teacher det confidence is high (already adapted).
                if len(pseudo_inst[0]) > 0:
                    mean_det_score = float(pseudo_inst[0].scores.mean().item())
                else:
                    mean_det_score = 0.0
                det_gate = mean_det_score < self.per_task_gate_det_thresh
                # Seg gate: skip if teacher seg confidence is high.
                seg_probs_gate = teacher_sem_results.float().softmax(dim=1)
                mean_seg_conf = float(seg_probs_gate.max(dim=1)[0].mean().item())
                seg_gate = mean_seg_conf < self.per_task_gate_seg_thresh

        # Diagnostic-only: score the spatial priors CT-CR could use against
        # semantic GT. Runs before CTPV so the box set is unfiltered, and is
        # independent of it so the current CTPV-off configs can be measured.
        if (self.diag_jsonl and self.mask_diag_gt_root
                and teacher_sem_results is not None
                and len(pseudo_inst[0]) > 0):
            with torch.no_grad():
                self._diag_write({
                    "type": "masked_ctcr",
                    "iter": int(self.iter),
                    "file_name": batched_inputs[0].get("file_name"),
                    "keep_step": bool(keep_step),
                    "boxes": self._diag_masked_ctcr_stats(
                        pseudo_inst,
                        teacher_sem_results.float().softmax(dim=1),
                        batched_inputs[0].get("file_name"),
                    ),
                })

        # V3: cross-task pseudo-label verification.
        if self.ctpv_enabled and len(pseudo_inst[0]) > 0:
            with torch.no_grad():
                t_seg_probs_ctpv = F.interpolate(
                    teacher_sem_results.float(),
                    size=(
                        teacher_sem_results.shape[-2],
                        teacher_sem_results.shape[-1],
                    ),
                    mode="bilinear",
                    align_corners=False,
                ).softmax(dim=1)

                n_before_ctpv = len(pseudo_inst[0])
                model_image_size = [
                    int(pseudo_inst[0].image_size[0]),
                    int(pseudo_inst[0].image_size[1]),
                ]

                if self.diag_jsonl:
                    pseudo_inst, ctpv_records = self._ctpv_filter(
                        pseudo_inst,
                        t_seg_probs_ctpv,
                        return_stats=True,
                    )
                else:
                    pseudo_inst = self._ctpv_filter(
                        pseudo_inst,
                        t_seg_probs_ctpv,
                    )
                    ctpv_records = []

                n_after_ctpv = len(pseudo_inst[0])

                if self.diag_jsonl:
                    self._diag_write({
                        "type": "ctpv",
                        "iter": int(self.iter),
                        "file_name": batched_inputs[0].get("file_name"),
                        "image_id": batched_inputs[0].get("image_id"),
                        "input_height": batched_inputs[0].get("height"),
                        "input_width": batched_inputs[0].get("width"),
                        "model_image_size": model_image_size,
                        "threshold": float(self.ctpv_thresh),
                        "score_em": float(self.score_em),
                        "keep_step": bool(keep_step),
                        "det_gate": bool(det_gate),
                        "seg_gate": bool(seg_gate),
                        "dynamic_thresholds": [float(x) for x in self.thresholds],
                        "n_raw_teacher": int(self._diag_raw_teacher_n),
                        "n_after_dynamic_threshold": int(self._diag_after_dyn_n),
                        "n_before": int(n_before_ctpv),
                        "n_after": int(n_after_ctpv),
                        "n_rejected": int(
                            n_before_ctpv - n_after_ctpv
                        ),
                        "rejection_rate": float(
                            (n_before_ctpv - n_after_ctpv)
                            / max(n_before_ctpv, 1)
                        ),
                        "boxes": ctpv_records,
                    })

        elif self.ctpv_enabled and self.diag_jsonl:
            # Log zero-pseudo images as well. This is important for exact
            # GT recall calculations; otherwise images with no pseudo-boxes
            # would silently disappear from the diagnostic trace.
            self._diag_write({
                "type": "ctpv",
                "iter": int(self.iter),
                "file_name": batched_inputs[0].get("file_name"),
                "image_id": batched_inputs[0].get("image_id"),
                "input_height": batched_inputs[0].get("height"),
                "input_width": batched_inputs[0].get("width"),
                "model_image_size": [
                    int(pseudo_inst[0].image_size[0]),
                    int(pseudo_inst[0].image_size[1]),
                ],
                "threshold": float(self.ctpv_thresh),
                "score_em": float(self.score_em),
                "keep_step": bool(keep_step),
                "det_gate": bool(det_gate),
                "seg_gate": bool(seg_gate),
                "dynamic_thresholds": [float(x) for x in self.thresholds],
                "n_raw_teacher": int(self._diag_raw_teacher_n),
                "n_after_dynamic_threshold": int(self._diag_after_dyn_n),
                "n_before": 0,
                "n_after": 0,
                "n_rejected": 0,
                "rejection_rate": 0.0,
                "boxes": [],
            })

        # 2. Student full forward (backbone -> heads) to get everything we need
        #    in one pass. With CTCMT_STRONG_AUG_STUDENT the student consumes the
        #    strong view while every teacher/anchor path keeps the weak one.
        images = self.student.preprocess_image(
            batched_inputs, strong_aug=self.strong_aug_student
        )
        features = self.student.backbone(images.tensor)

        loss_dict = {}

        if not self.seg_only and det_gate and len(pseudo_inst[0]) > 0:
            # ---- Det consistency: proposal + roi head standard losses
            # on teacher's pseudo boxes.
            gt = self._to_gt_instances(pseudo_inst)
            with EventStorage(self.iter):
                proposals, prop_losses = self.student.proposal_generator(images, features, gt)
                _, det_losses = self.student.roi_heads(images, features, proposals, gt)
            for k, v in prop_losses.items():
                loss_dict[f"det/{k}"] = self.weight_det * v
            for k, v in det_losses.items():
                loss_dict[f"det/{k}"] = self.weight_det * v

        # ---- Sem-seg soft-CE consistency (CoTTA-style).
        if not self.det_only and hasattr(self.student, "sem_seg_head"):
            s_seg_logits, _ = self.student.sem_seg_head(features, None)
        else:
            s_seg_logits = None
        if not self.det_only and seg_gate and s_seg_logits is not None:
            anchor_seg_probs = None
            teacher_seg_probs_full = F.interpolate(
                teacher_sem_results.float(), size=s_seg_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            ).softmax(dim=1)
            # CoTTA-style aug-average — trigger on anchor confidence (default)
            # or on TEACHER entropy (E3, addresses CoTTA-1 vulnerability).
            if self.seg_aug_enabled:
                with torch.no_grad():
                    if self.aug_trigger_teacher_entropy:
                        K = teacher_seg_probs_full.shape[1]
                        t_H = -(teacher_seg_probs_full.clamp_min(1e-8) *
                                teacher_seg_probs_full.clamp_min(1e-8).log()).sum(dim=1).mean()
                        norm_H = float(t_H.item()) / math.log(K)
                        trigger = norm_H > self.aug_teacher_entropy_thresh
                    else:
                        anchor_feats = self.anchor.backbone(images.tensor)
                        a_logits, _ = self.anchor.sem_seg_head(anchor_feats, None)
                        a_probs = a_logits.float().softmax(dim=1)
                        anchor_seg_probs = a_probs
                        conf = a_probs.max(dim=1)[0].mean()
                        trigger = float(conf.item()) < self.seg_aug_conf_thresh
                    if trigger:
                        aug_probs = self._teacher_aug_avg_seg_probs(images.tensor)
                        aug_probs = F.interpolate(
                            aug_probs, size=s_seg_logits.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )
                        teacher_seg_probs_full = aug_probs
                        self._seg_aug_fired += 1
                self._seg_steps += 1
            s_seg_log_probs = F.log_softmax(s_seg_logits.float(), dim=1)
            per_pixel_ce = -(teacher_seg_probs_full.detach() * s_seg_log_probs).sum(dim=1)

            # Multiplicative per-pixel weights; all-ones reproduces the plain mean.
            pixel_w = None
            if self.entropy_weighted_ce:
                # E2: down-weight uncertain pixels by (1 - normalized entropy).
                K = teacher_seg_probs_full.shape[1]
                with torch.no_grad():
                    tp = teacher_seg_probs_full.clamp_min(1e-8)
                    t_H_map = -(tp * tp.log()).sum(dim=1)                # (B, H, W)
                    ent_w = (1.0 - t_H_map / math.log(K)).clamp_min(0.0)
                pixel_w = ent_w if pixel_w is None else pixel_w * ent_w
            if self.class_balanced_ce:
                with torch.no_grad():
                    marg = self._update_class_marginal(teacher_seg_probs_full)
                    inv = marg.clamp_min(1e-6).pow(-self.class_balance_beta)
                    cb_w = inv[teacher_seg_probs_full.argmax(dim=1)]
                    cb_w = cb_w / cb_w.mean().clamp_min(1e-6)
                pixel_w = cb_w if pixel_w is None else pixel_w * cb_w

            if self.seg_det_veto and len(pseudo_inst[0]) > 0:
                # Inverted CTPV: trust the detector over the seg teacher inside
                # a confident box, by muting the pixels where they disagree.
                with torch.no_grad():
                    inst_v = pseudo_inst[0]
                    Hs, Ws = per_pixel_ce.shape[-2:]
                    ih, iw = inst_v.image_size
                    vx, vy = Ws / max(iw, 1), Hs / max(ih, 1)
                    t_arg = teacher_seg_probs_full.argmax(dim=1)
                    veto = torch.ones_like(per_pixel_ce)
                    v_scores = inst_v.scores.tolist()
                    v_classes = inst_v.pred_classes.long().tolist()
                    for jj, (bx1, by1, bx2, by2) in enumerate(
                            inst_v.pred_boxes.tensor.tolist()):
                        if v_scores[jj] < self.seg_det_veto_score:
                            continue
                        cc = v_classes[jj]
                        if not (0 <= cc < len(_DET_TO_SEG_CLASS_CITYSCAPES)):
                            continue
                        sc = _DET_TO_SEG_CLASS_CITYSCAPES[cc]
                        if sc >= teacher_seg_probs_full.shape[1]:
                            continue
                        a1 = max(int(round(bx1 * vx)), 0)
                        b1 = max(int(round(by1 * vy)), 0)
                        a2 = min(int(round(bx2 * vx)), Ws)
                        b2 = min(int(round(by2 * vy)), Hs)
                        if a2 <= a1 or b2 <= b1:
                            continue
                        sub = veto[0, b1:b2, a1:a2]
                        sub[t_arg[0, b1:b2, a1:a2] != sc] = self.seg_det_veto_weight
                pixel_w = veto if pixel_w is None else pixel_w * veto

            if pixel_w is None:
                loss_seg = per_pixel_ce.mean()
            else:
                loss_seg = (pixel_w * per_pixel_ce).sum() / (pixel_w.sum() + 1e-6)
                if self.seg_scale_preserve:
                    # Keep the reweighted gradient DIRECTION but restore the
                    # unweighted magnitude, so reweighting cannot silently
                    # change segmentation's share of the shared-trunk gradient.
                    with torch.no_grad():
                        scale = per_pixel_ce.mean() / loss_seg.clamp_min(1e-8)
                    loss_seg = loss_seg * scale
            loss_dict["seg/soft_ce"] = self.weight_seg * loss_seg

            # Mode-covering KL(anchor || student) on the predicted class
            # marginal: the student cannot drop a class the source still uses.
            if self.anchor_marginal_weight > 0:
                if anchor_seg_probs is None:
                    with torch.no_grad():
                        a_feats = self.anchor.backbone(images.tensor)
                        a_log, _ = self.anchor.sem_seg_head(a_feats, None)
                        anchor_seg_probs = a_log.float().softmax(dim=1)
                q = anchor_seg_probs.mean(dim=(0, 2, 3)).detach().clamp_min(1e-8)
                p = s_seg_logits.float().softmax(dim=1).mean(dim=(0, 2, 3)).clamp_min(1e-8)
                loss_dict["seg/anchor_marginal"] = (
                    self.anchor_marginal_weight * (q * (q.log() - p.log())).sum()
                )

            # Reliability signal for adaptive routing: how far the seg teacher
            # has drifted from the frozen source anchor. The anchor forward is
            # already paid for above, so this costs one argmax comparison.
            if self.adaptive_routing and anchor_seg_probs is not None:
                with torch.no_grad():
                    a = F.interpolate(
                        anchor_seg_probs, size=teacher_seg_probs_full.shape[-2:],
                        mode="bilinear", align_corners=False,
                    ).argmax(dim=1)
                    agree = float((teacher_seg_probs_full.argmax(dim=1) == a)
                                  .float().mean().item())
                self._update_route_lambda(agree)
        else:
            teacher_seg_probs_full = None

        # ---- Cross-task contrastive (CT-CL). Requires both branches active.
        if (self.ctcl_enabled and not self.det_only and not self.seg_only
                and det_gate and seg_gate
                and len(pseudo_inst[0]) > 0
                and teacher_seg_probs_full is not None):
            loss_ctcl = self._ctcl_loss(features, pseudo_inst, teacher_seg_probs_full)
            loss_dict["ctcl"] = self._cl_boost * self.weight_ctcl * loss_ctcl

        # ---- Cross-task consistency regularizer (CT-CR).
        if (self.weight_ctcr > 0 and not self.det_only and not self.seg_only
                and (det_gate or seg_gate) and len(pseudo_inst[0]) > 0):
            loss_ctcr = self._ctcr_loss(
                s_seg_logits,
                pseudo_inst,
                teacher_seg_probs_full,
            )
            if loss_ctcr is not None:
                loss_dict["ctcr"] = self._cl_boost * self.weight_ctcr * loss_ctcr

        # V4: cross-task prototype anchor loss.
        if (self.proto_anchor and not self.det_only and not self.seg_only
                and len(pseudo_inst[0]) > 0):
            proto_loss = self._proto_anchor_update_and_loss(
                features, pseudo_inst, teacher_seg_probs_full)
            if float(proto_loss.item()) > 0:
                loss_dict["proto"] = self.proto_weight * proto_loss

        # 3. Backward + step.
        fisher = None
        grad_diag = {}
        if loss_dict:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                grad_diag = self._backward_and_combine(loss_dict)
            except RuntimeError as exc:
                # Never let gradient surgery kill a multi-hour stream; degrade to
                # the joint update and count it so the run stays auditable.
                self._grad_fallbacks += 1
                if self._grad_fallbacks <= 5:
                    print(f"[CT-CMT-GRAD] WARNING iter={self.iter} surgery failed "
                          f"({exc}); falling back to joint backward")
                self.optimizer.zero_grad(set_to_none=True)
                total = sum(loss_dict.values())
                if total.requires_grad:
                    total.backward()
            if self.fisher_restore:
                fisher = {}
                for nm, m in self.student.named_modules():
                    for np_name, p in m.named_parameters(recurse=False):
                        if np_name in ("weight", "bias") and p.grad is not None:
                            fisher[f"{nm}.{np_name}"] = p.grad.detach().pow(2)
            self.optimizer.step()

        # 4. EMA + stochastic restore.
        self._fr_has_grad = bool(loss_dict)
        self._update_teacher()
        self._stochastic_restore(fisher)

        if self.iter % 50 == 0:
            self._fx_flush_log()
            self._fr_flush_log()
            summary = " ".join(f"{k}={float(v.detach()):.3f}" for k, v in loss_dict.items())
            tag = ""
            if self.per_task_gate:
                tag = f" det_gate={det_gate} seg_gate={seg_gate}"
            # thr exposes whether the dynamic threshold is pinned at THRESHOLD_MAX,
            # which caps how many pseudo-labels can ever enter the detector.
            _t = self.thresholds
            thr = f" thr={min(_t):.3f}/{sum(_t)/len(_t):.3f}/{max(_t):.3f}" if _t else ""
            aug = (f" segaug={self._seg_aug_fired}/{self._seg_steps}"
                   if self.seg_aug_enabled else "")
            rt = ""
            if self.conflict_mode in ("aux_head_only", "resnet_route"):
                rt = f" lam={self._route_lambda:.2f}"
                if self.conflict_mode == "resnet_route":
                    rt += " route=resnet"
                if self.adaptive_routing and self._seg_agree_ema is not None:
                    rt += (f" agree={self._seg_agree_ema:.4f}"
                           f"/{self._seg_agree_max:.4f}")
            print(f"[CT-CMT-MTL] iter={self.iter} score_em={self.score_em:.3f} "
                  f"n_pseudo={len(pseudo_inst[0])}{thr}{aug}{rt}{tag} {summary}")
            s = self._conflict_stats
            if s["both"] > 0:
                n = s["both"]
                if self.conflict_mode == "task_coco":
                    print(
                        f"[CT-CMT-COCO] iter={self.iter} "
                        f"cos_det_seg={grad_diag.get('cos', float('nan')):.4f} "
                        f"cos_mean={s['cos_sum'] / n:.4f} "
                        f"conflict_rate={s['conflicts'] / n:.4f} "
                        f"g_det={grad_diag.get('g_det', float('nan')):.5f} "
                        f"g_seg={grad_diag.get('g_seg', float('nan')):.5f} "
                        f"g_reg={grad_diag.get('g_reg', float('nan')):.5f} "
                        f"w_det={grad_diag.get('w_det', float('nan')):.4f} "
                        f"w_seg={grad_diag.get('w_seg', float('nan')):.4f} "
                        f"c_det={grad_diag.get('c_det', float('nan')):.4f} "
                        f"c_seg={grad_diag.get('c_seg', float('nan')):.4f} "
                        f"w_det_mean={s['w_det_sum'] / n:.4f} "
                        f"w_seg_mean={s['w_seg_sum'] / n:.4f} "
                        f"c_det_mean={s['c_det_sum'] / n:.4f} "
                        f"c_seg_mean={s['c_seg_sum'] / n:.4f} "
                        f"GC_c={self.task_coco_gc_c:.3f} "
                        f"fallbacks={self._grad_fallbacks} "
                        f"n_steps={n}"
                    )
                else:
                    print(f"[CT-CMT-GRAD] iter={self.iter} mode={self.conflict_mode} "
                      f"cos={grad_diag.get('cos', float('nan')):.4f} "
                      f"cos_mean={s['cos_sum'] / n:.4f} "
                      f"conflict_rate={s['conflicts'] / n:.4f} "
                      f"applied_rate={s['projected'] / n:.4f} "
                      f"g_det={grad_diag.get('g_det', float('nan')):.5f} "
                      f"g_aux={grad_diag.get('g_aux', float('nan')):.5f} "
                      f"g_aux_proj={grad_diag.get('g_aux_proj', float('nan')):.5f} "
                      f"g_det_R={grad_diag.get('g_det_resnet', float('nan')):.5f} "
                      f"g_aux_R={grad_diag.get('g_aux_resnet', float('nan')):.5f} "
                      f"g_aux_R_applied={grad_diag.get('g_aux_resnet_applied', float('nan')):.5f} "
                      f"g_det_FPN={grad_diag.get('g_det_fpn', float('nan')):.5f} "
                      f"g_aux_FPN={grad_diag.get('g_aux_fpn', float('nan')):.5f} "
                      f"g_aux_FPN_applied={grad_diag.get('g_aux_fpn_applied', float('nan')):.5f} "
                      f"gamma={grad_diag.get('gamma', float('nan')):.4f} "
                      f"gamma_mean={s['gamma_sum'] / n:.4f} "
                      f"w_det={grad_diag.get('w_det', float('nan')):.4f} "
                          f"w_det_mean={s['w_det_sum'] / n:.4f} "
                          f"fallbacks={self._grad_fallbacks} "
                          f"n_steps={n}")
        if grad_diag and any(k.startswith("cos_det_") for k in grad_diag):
            comps = " ".join(
                f"{k}={grad_diag[k]:.4f}" for k in sorted(grad_diag) if k.startswith("cos_det_")
            )
            norms = " ".join(
                f"{k}={grad_diag[k]:.6f}" for k in sorted(grad_diag) if k.startswith("gnorm_")
            )
            blocks = " ".join(
                f"blk_{k}={v:.4f}" for k, v in sorted(grad_diag.get("blocks", {}).items())
            )
            print(f"[CT-CMT-GRADDIAG] iter={self.iter} {comps} {norms} {blocks}")

        # 5. Report TEACHER predictions for evaluation (no panoptic combine).
        with torch.no_grad():
            # Only skip the detector when the teacher genuinely has none (ST-S
            # specialist). A seg-only MTL teacher still has one, and its
            # detection AP is what measures seg adaptation's effect on the
            # shared trunk.
            if self.seg_only and not hasattr(self.teacher, "roi_heads"):
                # No detector, and the teacher must see the weak view even when
                # the student was trained on the strong one.
                t_images = self.teacher.preprocess_image(batched_inputs)
                t_sem, _ = self.teacher.sem_seg_head(
                    self.teacher.backbone(t_images.tensor), None
                )
                t_det = None
                image_sizes = t_images.image_sizes
            else:
                if pre_update_out is not None:
                    teacher_out = pre_update_out
                else:
                    teacher_out = self.teacher.inference(batched_inputs, do_postprocess=False)
                if isinstance(teacher_out, tuple) and len(teacher_out) == 2:
                    t_det, t_sem = teacher_out
                else:
                    t_det, t_sem = teacher_out, None
                image_sizes = images.image_sizes
        processed = []
        for i, (inp, image_size) in enumerate(zip(batched_inputs, image_sizes)):
            H = inp.get("height", image_size[0])
            W = inp.get("width", image_size[1])
            out_i = {}
            if t_det is not None:
                out_i["instances"] = detector_postprocess(t_det[i], H, W)
            if t_sem is not None:
                out_i["sem_seg"] = sem_seg_postprocess(t_sem[i], image_size, H, W)
            processed.append(out_i)
        return processed
