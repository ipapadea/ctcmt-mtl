"""CT-CMT (Cross-Task Contrastive Mean Teacher) — MTL extension of
:class:`MeanTeacherYOLOXAdapterContrastive` (Moraiti et al. 2026).

New components relative to the prior single-task work:

  1. Segmentation branch — the wrapped detector must be a
     :class:`YOLOXMTLDetector` so it exposes ``seg_head`` alongside
     ``bbox_head``. Teacher's seg softmax is used to generate soft
     pseudo-labels for the student's seg head via a soft-CE consistency
     loss (:class:`MTLConsistencyLoss`).

  2. Cross-Task Contrastive Learning (CT-CL) — the object-level SupCon of
     the prior work is extended so that, for each teacher-detected bbox,
     both a *detection view* (RoIAlign at the bbox) and a *segmentation view*
     (bbox-region features weighted by the teacher's seg probability for the
     bbox's class) contribute. Cross-task pairs of the same object are pulled
     together; different-class pairs are pushed apart. See
     :class:`CrossTaskContrastiveLoss` for details.

  3. Cross-Task Consistency Regularizer (CT-CR) — bbox-region pixels in the
     student's seg prediction should agree with the teacher's detected class
     for that bbox. Cheap add-on, small weight.

  4. Task-Weighted Stochastic Restoration (optional) — instead of uniform
     Bernoulli restore, the per-parameter restore probability is scaled by
     recent parameter drift. Enabled via ``task_weighted_restore=True``.

Everything else — 5 augmented views of the student, teacher EMA (via
``mmtrack``'s teacher module built by the config), YOLOX consistency, SR —
is inherited from the prior single-task adapter's design and reproduced
verbatim here so this file is self-contained.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from mmengine.structures import InstanceData, PixelData
from mmtrack.structures import TrackDataSample

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter

# Cityscapes: det class index (ACDC order) -> seg trainId
# ('person','rider','car','train','motorcycle','bicycle','truck','bus')
_DET_TO_SEG_CLS_ACDC = (11, 12, 13, 16, 17, 18, 14, 15)


def _is_shared_trunk(name: str) -> bool:
    """True for backbone/neck params shared by both heads in YOLOX-MTL."""
    return name.startswith(("yolox.backbone.", "yolox.neck."))


def _dyn_thresholds(prev, per_class_mean, alpha, gamma, lo, hi):
    new = []
    for th, mean in zip(prev, per_class_mean):
        if mean > 0:
            th = gamma * th + (1.0 - gamma) * alpha * math.sqrt(mean)
        th = max(min(th, hi), lo)
        new.append(th)
    return new


@MODELS.register_module()
class MeanTeacherYOLOXMTLAdapterCTCMT(BaseAdapter):
    """CT-CMT adapter — MTL Mean Teacher with cross-task contrastive learning.

    Args:
        teacher: mmtrack teacher config (same as prior single-task adapter).
        fixed_source_model: keep a frozen source copy for restoration.
        stochastic_restoration: enable per-step SR.
        rst_prob: SR Bernoulli probability. Paper default 0.01.
        task_weighted_restore: scale SR probability by recent parameter drift.
        optim_wrapper: optimizer wrapper cfg (matches prior work).
        optim_steps: number of student updates per input image (paper: 5,
            i.e. the same augmented view is shown 5 times).
        views: number of strong-augmentation views per input image.
        pipeline / teacher_pipeline / student_pipeline: image transforms
            (same interface as the prior single-task adapter).
        mtl_consistency_loss: config for :class:`MTLConsistencyLoss`.
        ct_cl_loss: config for :class:`CrossTaskContrastiveLoss`.
        ct_cr_weight: weight of the cross-task consistency regularizer
            (seg-inside-bbox class agreement). Set 0 to disable.
        spatial_scales: PAFPN spatial scales (P3/P4/P5 -> 1/8, 1/16, 1/32).
        seg_num_classes: number of seg classes (needed for the CT-CR loss
            because we compare student seg logits vs. detected class ids).
        det_conf_threshold: min teacher detection score to keep a bbox as a
            pseudo-label (used for CT-CL and CT-CR).
        filter_pseudo_labels: same as prior work — YOLOX NMS score threshold
            passed to ``predict_by_feat``.
    """

    def __init__(
        self,
        teacher: Optional[dict] = None,
        fixed_source_model: bool = True,
        stochastic_restoration: bool = True,
        rst_prob: float = 0.01,
        task_weighted_restore: bool = False,
        optim_wrapper: Optional[dict] = None,
        optim_steps: int = 5,
        views: int = 2,
        pipeline: Optional[list[dict]] = None,
        teacher_pipeline: Optional[list[dict]] = None,
        student_pipeline: Optional[list[dict]] = None,
        mtl_consistency_loss: Optional[dict] = None,
        ct_cl_loss: Optional[dict] = None,
        ct_cr_weight: float = 0.1,
        spatial_scales: Tuple[float, float, float] = (1 / 8, 1 / 16, 1 / 32),
        seg_num_classes: int = 19,
        det_conf_threshold: float = 0.7,
        filter_pseudo_labels: float = 0.7,
        # Score-EM gate (METHOD.md §6)
        score_em_init: float = 0.5,
        score_gamma: float = 0.7,
        score_thresh: float = 1.4,
        # CTPV (METHOD.md §5.5)
        ctpv_enabled: bool = False,
        ctpv_thresh: float = 0.3,
        det_to_seg_cls: Optional[Tuple[int, ...]] = None,
        # STR shared-trunk restore factor η (METHOD.md §5)
        backbone_rst_factor: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.teacher = None
        self.teacher_cfg = teacher

        self.optim_wrapper = None
        self.optim_wrapper_cfg = optim_wrapper
        self.optim_steps = int(optim_steps)

        self.mtl_loss_cfg = mtl_consistency_loss or dict(type="MTLConsistencyLoss")
        self.ct_cl_cfg = ct_cl_loss or dict(type="CrossTaskContrastiveLoss")
        self.mtl_loss = MODELS.build(self.mtl_loss_cfg)
        self.ct_cl = MODELS.build(self.ct_cl_cfg)

        self.pipeline = Compose(pipeline) if pipeline else None
        self.teacher_pipeline = Compose(teacher_pipeline) if teacher_pipeline else None
        self.student_pipeline = Compose(student_pipeline) if student_pipeline else None

        self.views = int(views)
        self.stochastic_restoration = bool(stochastic_restoration)
        self.rst_prob = float(rst_prob)
        self.task_weighted_restore = bool(task_weighted_restore)
        self.fixed_source_model = bool(fixed_source_model)

        self.ct_cr_weight = float(ct_cr_weight)
        self.spatial_scales = tuple(float(s) for s in spatial_scales)
        self.seg_num_classes = int(seg_num_classes)
        self.det_conf_threshold = float(det_conf_threshold)
        self.filter_pseudo_labels = float(filter_pseudo_labels)

        # Score-EM gate
        self.score_em = float(score_em_init)
        self.score_gamma = float(score_gamma)
        self.score_thresh = float(score_thresh)

        # CTPV
        self.ctpv_enabled = bool(ctpv_enabled)
        self.ctpv_thresh = float(ctpv_thresh)
        self.det_to_seg_cls = (
            tuple(int(x) for x in det_to_seg_cls)
            if det_to_seg_cls is not None
            else None
        )

        # STR backbone factor
        self.backbone_rst_factor = float(backbone_rst_factor)

        # Bookkeeping for task-weighted restore
        self._drift_ema: Dict[str, Tensor] = {}
        self._drift_ema_alpha = 0.9

    # ------------------------------------------------------------------
    # Source-model bookkeeping (matches the prior single-task adapter)
    # ------------------------------------------------------------------
    def _init_source_model_state(self, model) -> None:
        super()._init_source_model_state(model)

        if self.optim_wrapper_cfg is not None:
            self.optim_wrapper = build_optim_wrapper(model, self.optim_wrapper_cfg)
            self.optim_wrapper_state = deepcopy(self.optim_wrapper.state_dict())

        if self.teacher_cfg is not None:
            self.teacher_cfg["model"] = model
            self.teacher = MODELS.build(self.teacher_cfg)
            self.teacher_model_state = deepcopy(self.teacher.state_dict())

        if self.fixed_source_model:
            self.fixed_model = deepcopy(model)

    def _restore_source_model_state(self, model) -> None:
        super()._restore_source_model_state(model)
        if self.teacher is not None:
            self.teacher.load_state_dict(self.teacher_model_state, strict=True)

    def _reset_optimizer(self) -> None:
        if self.optim_wrapper is not None:
            self.optim_wrapper.load_state_dict(self.optim_wrapper_state)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _forward_outputs(
        self,
        model_detector,
        img: Tensor,
        data_samples: List[TrackDataSample],
        rescale: bool = False,
    ) -> Dict[str, Any]:
        """Run a YOLOXMTLDetector-style detector on img and return raw head
        outputs, PAFPN features, and post-processed det predictions.

        Works with either :class:`YOLOXMTLDetector` (has ``.yolox``, ``.seg_head``)
        or the plain single-task mmdet YOLOX (in which case seg outputs are None).

        Args:
            rescale: if True, ``det_predictions`` are rescaled to the original
                image resolution (via ``data_samples[i].metainfo['scale_factor']``)
                so they can be reported to eval metrics. If False, they stay
                in the detector's input pixel space (needed for CT-CL RoIAlign
                and CT-CR bbox\u2192seg-grid mapping).
        """
        if hasattr(model_detector, "yolox"):
            core = model_detector.yolox
            seg_head = model_detector.seg_head
        else:
            core = model_detector
            seg_head = None

        feats = core.extract_feat(img)
        det_head_outs = core.bbox_head(feats)  # (cls_list, reg_list, obj_list)

        batch_img_metas = [d.metainfo for d in data_samples]
        # Override the head's default score threshold with our pseudo-label
        # threshold (mirrors the single-task contrastive adapter).
        cfg = deepcopy(core.bbox_head.test_cfg)
        if not rescale:
            cfg["score_thr"] = self.filter_pseudo_labels
        predictions = core.bbox_head.predict_by_feat(
            *det_head_outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            cfg=cfg,
        )

        det_dict = dict(
            cls_score=det_head_outs[0],
            bbox_pred=det_head_outs[1],
            objectness=det_head_outs[2],
        )

        seg_logits = seg_head(feats) if seg_head is not None else None

        return dict(
            feats=feats,
            det_outs=det_dict,
            det_predictions=predictions,
            seg_logits=seg_logits,
        )

    @staticmethod
    def _expand_view(head_outs: Dict[str, list], views: int) -> Dict[str, list]:
        """Repeat each per-level tensor along the batch dim ``views`` times."""
        return {
            k: [t.repeat_interleave(views, dim=0) for t in v]
            for k, v in head_outs.items()
        }

    # ------------------------------------------------------------------
    # Teacher-generated pseudo-boxes -> lists per image
    # ------------------------------------------------------------------
    def _split_teacher_predictions(
        self,
        predictions: List[InstanceData],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        boxes_per_image: List[Tensor] = []
        classes_per_image: List[Tensor] = []
        for pred in predictions:
            keep = pred.scores >= self.det_conf_threshold
            boxes_per_image.append(pred.bboxes[keep].detach())
            classes_per_image.append(pred.labels[keep].detach().long())
        return boxes_per_image, classes_per_image

    # ------------------------------------------------------------------
    # Cross-Task Consistency Regularizer (CT-CR)
    # ------------------------------------------------------------------
    def _ct_cr_loss(
        self,
        student_seg_logits: Tensor,
        boxes_per_image: List[Tensor],
        classes_per_image: List[Tensor],
    ) -> Tensor:
        """Bbox-region seg CE: pixels inside a teacher-detected bbox should
        be classified as the bbox's class in the student's seg output."""
        B, C, H, W = student_seg_logits.shape
        # Build a (B, H, W) target with 255 (ignore) everywhere except inside
        # teacher-detected boxes, where the target is the bbox class.
        target = torch.full(
            (B, H, W), 255, dtype=torch.long, device=student_seg_logits.device
        )
        # Boxes are in input-image pixel space; convert to seg-logit grid.
        seg_hw = torch.tensor([H, W], dtype=torch.float, device=target.device)
        for i, (boxes, classes) in enumerate(zip(boxes_per_image, classes_per_image)):
            if boxes.numel() == 0:
                continue
            # Assume input->seg-logits scale = seg_hw / input_hw (approx).
            # We rescale via ratio of grid to first bbox's max coord if
            # metainfo is not accessible.
            x1 = boxes[:, 0].clamp(min=0)
            y1 = boxes[:, 1].clamp(min=0)
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            # Normalize by the maximum bbox coordinate + margin so we always
            # remain in-bounds even when the seg-logits grid is smaller than
            # the input. This is an approximation; if you need pixel-precise
            # placement, pass metainfo through.
            # In practice the seg logits are at stride 8, so we scale down 8x.
            sx = (float(W) / max(float(x2.max().item()), 1.0)) * 0.999
            sy = (float(H) / max(float(y2.max().item()), 1.0)) * 0.999
            x1i = (x1 * sx).long().clamp(0, W - 1)
            x2i = (x2 * sx).long().clamp(0, W)
            y1i = (y1 * sy).long().clamp(0, H - 1)
            y2i = (y2 * sy).long().clamp(0, H)
            for j in range(boxes.size(0)):
                target[i, y1i[j]:y2i[j], x1i[j]:x2i[j]] = int(classes[j].item())

        if (target != 255).sum() == 0:
            return torch.zeros((), device=student_seg_logits.device)

        return F.cross_entropy(student_seg_logits, target, ignore_index=255)

    # ------------------------------------------------------------------
    # CTPV: cross-task pseudo-label verification (METHOD.md §5.5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ctpv_filter(
        self,
        boxes_per_image: List[Tensor],
        classes_per_image: List[Tensor],
        teacher_seg_probs: Tensor,
        img_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Reject box b of class c when the teacher seg head assigns < ctpv_thresh
        majority vote to σ(c) over the box interior."""
        _, K_seg, Hs, Ws = teacher_seg_probs.shape
        filtered_boxes, filtered_classes = [], []
        for i, (boxes, classes) in enumerate(zip(boxes_per_image, classes_per_image)):
            if boxes.numel() == 0:
                filtered_boxes.append(boxes)
                filtered_classes.append(classes)
                continue
            seg_probs_i = teacher_seg_probs[i]
            if img_shapes is not None and i < len(img_shapes):
                ih, iw = float(img_shapes[i][0]), float(img_shapes[i][1])
            else:
                iw = float(boxes[:, 2].max().item()) / 0.999
                ih = float(boxes[:, 3].max().item()) / 0.999
            sx = Ws / max(iw, 1.0)
            sy = Hs / max(ih, 1.0)
            keep = []
            for j in range(boxes.size(0)):
                c = int(classes[j].item())
                seg_c = (
                    self.det_to_seg_cls[c]
                    if self.det_to_seg_cls is not None and c < len(self.det_to_seg_cls)
                    else c
                )
                if seg_c >= K_seg:
                    keep.append(True)
                    continue
                x1, y1, x2, y2 = boxes[j].tolist()
                x1i = max(int(round(x1 * sx)), 0)
                y1i = max(int(round(y1 * sy)), 0)
                x2i = min(int(round(x2 * sx)), Ws)
                y2i = min(int(round(y2 * sy)), Hs)
                if x2i <= x1i or y2i <= y1i:
                    keep.append(False)
                    continue
                region = seg_probs_i[:, y1i:y2i, x1i:x2i]
                agreement = float((region.argmax(dim=0) == seg_c).float().mean())
                keep.append(agreement >= self.ctpv_thresh)
            keep_t = torch.tensor(keep, dtype=torch.bool, device=boxes.device)
            filtered_boxes.append(boxes[keep_t])
            filtered_classes.append(classes[keep_t])
        return filtered_boxes, filtered_classes

    # ------------------------------------------------------------------
    # Stochastic Restoration — STR backbone split (METHOD.md §5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _stochastic_restore(self, model) -> None:
        if not self.stochastic_restoration or self.fixed_model is None:
            return
        src_state = self.fixed_model.state_dict()
        student_state = model.state_dict()
        for name, p in student_state.items():
            if not p.dtype.is_floating_point:
                continue
            src = src_state.get(name)
            if src is None or src.shape != p.shape:
                continue
            src = src.to(p.device)

            if self.task_weighted_restore:
                drift = (p.detach() - src).abs()
                ema = self._drift_ema.get(name)
                if ema is None:
                    ema = drift.clone()
                else:
                    ema.mul_(self._drift_ema_alpha).add_(
                        drift, alpha=1.0 - self._drift_ema_alpha
                    )
                self._drift_ema[name] = ema
                gate = ema / (ema.mean() + 1e-8)
                probs = torch.clamp(gate * self.rst_prob, max=0.5)
            else:
                # STR: shared trunk (backbone+neck) restored more aggressively
                base = (
                    self.rst_prob * self.backbone_rst_factor
                    if _is_shared_trunk(name)
                    else self.rst_prob
                )
                probs = torch.full_like(p, base)

            mask = (torch.rand_like(p) < probs).float()
            p.mul_(1.0 - mask).add_(src * mask)

    # ------------------------------------------------------------------
    # Core adaptation step (one gradient update)
    # ------------------------------------------------------------------
    def _adapt(
        self,
        model: torch.nn.Module,
        teacher_img: Tensor,
        student_imgs: Tensor,
        teacher_data_samples: List[TrackDataSample],
        student_data_samples: List[TrackDataSample],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """One CT-CMT test-time step.

        Runs a teacher forward (no grad, input-space predictions), a student
        forward on the augmented views, computes MTL consistency + CT-CL +
        CT-CR, and does a single optimizer step. Returns the teacher forward
        outputs dict so the caller can reuse them if needed.
        """
        with torch.no_grad():
            teacher_out = self._forward_outputs(
                self.teacher.module.detector,
                teacher_img,
                teacher_data_samples,
                rescale=False,
            )
        teacher_det_outs = self._expand_view(teacher_out["det_outs"], self.views)
        teacher_seg_logits = teacher_out["seg_logits"]
        teacher_seg_probs = (
            F.softmax(teacher_seg_logits.detach(), dim=1)
            if teacher_seg_logits is not None else None
        )

        # === Score-EM gate (METHOD.md §6) ===
        all_det_scores = torch.cat(
            [p.scores[p.scores > 0.1] for p in teacher_out["det_predictions"]
             if p.scores.numel() > 0],
            dim=0,
        )
        if all_det_scores.numel() == 0:
            return teacher_out
        mean_score = float(all_det_scores.mean().item())
        ratio = mean_score / max(self.score_em, 1e-6)
        self.score_em = self.score_gamma * self.score_em + (1.0 - self.score_gamma) * mean_score
        if ratio > self.score_thresh or (1.0 / max(ratio, 1e-6)) > self.score_thresh:
            self._stochastic_restore(model)
            return teacher_out

        boxes_per_image, classes_per_image = self._split_teacher_predictions(
            teacher_out["det_predictions"]
        )

        # === CTPV (METHOD.md §5.5) ===
        if self.ctpv_enabled and teacher_seg_probs is not None:
            img_shapes = [
                tuple(d.metainfo.get("img_shape", (0, 0))[:2])
                for d in teacher_data_samples
            ]
            boxes_per_image, classes_per_image = self._ctpv_filter(
                boxes_per_image, classes_per_image, teacher_seg_probs, img_shapes,
            )
        boxes_per_view = [b for b in boxes_per_image for _ in range(self.views)]
        classes_per_view = [c for c in classes_per_image for _ in range(self.views)]

        student_out = self._forward_outputs(
            model.detector, student_imgs, student_data_samples, rescale=False,
        )
        student_det_outs = student_out["det_outs"]
        student_seg_logits = student_out["seg_logits"]

        cons = self.mtl_loss(
            student_det_outs=student_det_outs,
            teacher_det_outs=teacher_det_outs,
            student_seg_logits=student_seg_logits,
            teacher_seg_logits=teacher_seg_logits,
        )

        ct_cl = self.ct_cl(
            feature_maps=student_out["feats"],
            boxes_per_image=boxes_per_image,
            classes_per_image=classes_per_image,
            spatial_scales=self.spatial_scales,
            teacher_seg_probs=teacher_seg_probs,
        )

        if (self.ct_cr_weight > 0
                and student_seg_logits is not None
                and any(b.numel() for b in boxes_per_view)):
            ct_cr = self.ct_cr_weight * self._ct_cr_loss(
                student_seg_logits, boxes_per_view, classes_per_view
            )
        else:
            ct_cr = torch.zeros((), device=student_imgs.device)

        total_loss = cons["total"] + ct_cl + ct_cr

        self.optim_wrapper.zero_grad()
        total_loss.backward()
        self.optim_wrapper.step()

        return teacher_out

    # ------------------------------------------------------------------
    # Seg-prediction bookkeeping
    # ------------------------------------------------------------------
    @staticmethod
    def _seg_logits_to_ori(
        seg_logits: Tensor,
        teacher_metainfo: Dict[str, Any],
        ori_hw: Tuple[int, int],
    ) -> Tensor:
        """Rescale (B, C, H_seg, W_seg) logits from the seg-head grid back to
        original image resolution. Handles the Resize\u2192Pad chain applied by
        the teacher pipeline: upsample to padded size, crop to img_shape,
        interpolate to ori_shape. Returns argmax (H_ori, W_ori) long."""
        pad_hw = teacher_metainfo.get("pad_shape", None)
        img_hw = teacher_metainfo.get("img_shape", None)
        if pad_hw is None or img_hw is None:
            up = F.interpolate(
                seg_logits, size=ori_hw, mode="bilinear", align_corners=False,
            )
            return up[0].argmax(dim=0).long()

        pad_h, pad_w = int(pad_hw[0]), int(pad_hw[1])
        img_h, img_w = int(img_hw[0]), int(img_hw[1])
        ori_h, ori_w = int(ori_hw[0]), int(ori_hw[1])
        up = F.interpolate(
            seg_logits, size=(pad_h, pad_w), mode="bilinear", align_corners=False,
        )
        up = up[..., :img_h, :img_w]
        up = F.interpolate(
            up, size=(ori_h, ori_w), mode="bilinear", align_corners=False,
        )
        return up[0].argmax(dim=0).long()

    # ------------------------------------------------------------------
    # Outer entry point (called by AdaptiveDetector.predict per sample)
    # ------------------------------------------------------------------
    def adapt(
        self,
        model: torch.nn.Module,
        img: Tensor,
        feats: Optional[List[Tensor]],
        data_sample: TrackDataSample,
        **kwargs,
    ) -> InstanceData:
        """Full per-sample TTA cycle.

        Steps:
            1. Reset to source on new sequence (episodic + frame_id==0).
            2. Build teacher and ``views`` student augmented copies via the
               configured pipelines.
            3. Run ``optim_steps`` gradient steps; EMA-update the teacher after
               each step.
            4. Stochastic Restoration (optional, once per input image).
            5. Final teacher forward with ``rescale=True`` to produce
               ``pred_det_instances`` and ``pred_sem_seg`` in the original
               image coordinate frame.
        """
        frame_id = data_sample.metainfo.get("frame_id", -1)
        if self.with_episodic and frame_id == 0:
            self.reset(model)
        if self.source_model_state is None:
            self._init_source_model_state(model)

        # Build a fresh results dict from the sample; teacher/student pipelines
        # apply Resize+Pad+PackTrackInputs on top of a preloaded image.
        results = dict(
            img_path=data_sample.img_path,
            instances=getattr(data_sample, "instances", []),
        )
        if self.pipeline is not None:
            results = self.pipeline(results)

        teacher_results = self.teacher_pipeline(deepcopy(results))
        teacher_img = teacher_results["inputs"]["img"].to(img)
        teacher_data_samples = [teacher_results["data_samples"]]

        student_imgs: List[Tensor] = []
        student_data_samples: List[TrackDataSample] = []
        for _ in range(self.views):
            sr = self.student_pipeline(deepcopy(results))
            student_imgs.append(sr["inputs"]["img"])
            student_data_samples.append(sr["data_samples"])
        student_imgs_t = torch.cat(student_imgs, dim=0).to(img)

        with torch.enable_grad():
            model.requires_grad_(True)
            model.train(True)
            for _ in range(max(1, self.optim_steps)):
                self._adapt(
                    model,
                    teacher_img,
                    student_imgs_t,
                    teacher_data_samples,
                    student_data_samples,
                )
                self.teacher.update_parameters(model)
            if self.stochastic_restoration:
                self._stochastic_restore(model)

        self._reset_optimizer()

        # Final teacher forward for evaluation (rescaled to ori image space).
        model.eval()
        with torch.no_grad():
            final_out = self._forward_outputs(
                self.teacher.module.detector,
                teacher_img,
                teacher_data_samples,
                rescale=True,
            )

        seg_logits = final_out["seg_logits"]
        if seg_logits is not None:
            ori_hw = data_sample.metainfo.get(
                "ori_shape", tuple(teacher_img.shape[-2:])
            )
            seg_pred = self._seg_logits_to_ori(
                seg_logits, teacher_data_samples[0].metainfo, ori_hw
            )
            data_sample.pred_sem_seg = PixelData(sem_seg=seg_pred[None])

        preds = final_out["det_predictions"]
        return preds[0].clone() if preds else InstanceData()
