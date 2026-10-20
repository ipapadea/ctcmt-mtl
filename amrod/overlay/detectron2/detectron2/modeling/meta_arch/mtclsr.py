# MT + object-level contrastive learning + stochastic restoration
# (Moraiti et al., "Continual Test-Time Domain Adaptation for Object Detection via
# Contrastive Mean Teacher and Stochastic Restoration", EJAI 39(2), 2025).
#
# Same-source port of the official YOLOX implementation
#   shift-detection-tta / MeanTeacherYOLOXAdapterContrastive
#   shift-detection-tta / YOLOXConsistencyContrastiveLoss
# onto this repository's Panoptic-FPN R50 checkpoint (detection branch only).
# The adaptation recipe follows the official code; only the parts that are tied to
# the YOLOX head had to be re-expressed for Faster R-CNN (see "PORT" notes).
#
# Per image (batch size 1, no reset):
#   repeat STEPS times (the same VIEWS strong views every time):
#       teacher (weak view)  -> raw outputs + detections
#       student (strong views) -> raw outputs
#       loss = W_CONSISTENCY * MSE-consistency + W_CONTRASTIVE * sum_levels SupCon
#       SGD step on the student, then teacher = EMA(student)
#   Bernoulli stochastic restoration of the student once, after the last step.
#   Optimizer state (SGD momentum) is cleared after every image (official code).
# The reported prediction is the teacher output computed at the start of the LAST
# step (official code); TEST.MTCLSR.PREDICT="post" adds a pass after restoration.
#
# PORT notes (YOLOX -> Faster R-CNN):
#   * YOLOX consistency = MSE on dense (objectness, bbox, cls) head outputs.  Here:
#     MSE on RPN objectness logits, RPN anchor deltas, ROI-head class logits and
#     ROI-head box deltas, each averaged over all elements; ROI outputs are taken on
#     the teacher's proposals for both networks.  Raw logits/deltas, as in YOLOX.
#   * Contrastive features = 1x1 RoIAlign of FPN levels p3/p4/p5 (the three YOLOX
#     PAFPN scales have strides 8/16/32), L2-normalised, on the teacher's boxes with
#     score >= PSEUDO_SCORE_THR.  Anchors are the teacher's features and the contrast
#     set contains the teacher and every student view (SupConLoss contrast_mode="one").
#   * The student augmentations (ColorJitter, RandomGrayscale, GaussianBlur, Solarize,
#     RandomErasing) are re-implemented with torchvision on the resized input image.
import logging
import math
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import nn
from torchvision.ops import batched_nms, roi_align

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils.run_provenance import log_run_provenance

from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN

logger = logging.getLogger(__name__)

# (aspect_range, min_area_ratio, max_area_ratio) of the six RandomErasing ops that the
# official student pipeline applies together with probability 0.7.
_ERASE_OPS = (
    ((0.3, 3.3), 0.001, 0.005),
    ((0.1, 6.0), 0.001, 0.005),
    ((0.05, 8.0), 0.001, 0.005),
    ((0.05, 8.0), 0.0005, 0.0025),
    ((0.05, 8.0), 0.0005, 0.0025),
    ((0.05, 8.0), 0.0005, 0.001),
)


def _random_erase(img: torch.Tensor, aspect_range, min_ratio, max_ratio) -> torch.Tensor:
    _, H, W = img.shape
    for _ in range(10):
        area = H * W * (min_ratio + (max_ratio - min_ratio) * torch.rand(1).item())
        log_lo, log_hi = math.log(aspect_range[0]), math.log(aspect_range[1])
        aspect = math.exp(log_lo + (log_hi - log_lo) * torch.rand(1).item())
        h = int(round(math.sqrt(area * aspect)))
        w = int(round(math.sqrt(area / aspect)))
        if 0 < h < H and 0 < w < W:
            top = int(torch.randint(0, H - h + 1, (1,)).item())
            left = int(torch.randint(0, W - w + 1, (1,)).item())
            noise = torch.randint(0, 256, (img.shape[0], h, w), device=img.device, dtype=img.dtype)
            img[:, top:top + h, left:left + w] = noise
            return img
    return img


def strong_view(img_rgb_u8: torch.Tensor) -> torch.Tensor:
    """One strongly augmented view of a uint8 RGB CHW tensor (official student pipeline)."""
    img = img_rgb_u8.clone()
    if torch.rand(1).item() < 0.8:
        img = T.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)
    if torch.rand(1).item() < 0.2:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    if torch.rand(1).item() < 0.5:
        k = int(torch.randint(1, 4, (1,)).item()) * 2 + 1  # 3, 5, 7
        sigma = 0.1 + (2.0 - 0.1) * torch.rand(1).item()
        img = TF.gaussian_blur(img, [k, k], [sigma, sigma])
    if torch.rand(1).item() < 0.025:
        img = TF.solarize(img, 128)
    if torch.rand(1).item() < 0.7:
        img = img.clone(memory_format=torch.contiguous_format)  # grayscale returns an expanded view
        for aspect, lo, hi in _ERASE_OPS:
            img = _random_erase(img, aspect, lo, hi)
    return img


def supcon_teacher_anchor(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupConLoss(contrast_mode="one") of the official code.

    features: (N, V, D) L2-normalised; view 0 (the teacher) provides the anchors and
    all V views are the contrast set.  labels: (N,) pseudo classes.
    """
    N, V, _ = features.shape
    device = features.device
    mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)  # (V*N, D), view-major
    anchor = features[:, 0]
    logits = torch.matmul(anchor, contrast.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = mask.repeat(1, V)
    logits_mask = torch.ones_like(mask)
    logits_mask[torch.arange(N, device=device), torch.arange(N, device=device)] = 0
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    return -mean_log_prob_pos.mean()


@META_ARCH_REGISTRY.register()
class MTCLSR(nn.Module):
    @configurable
    def __init__(self, *, student: nn.Module, teacher: nn.Module, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg.TEST.MTCLSR
        self.steps = int(c.STEPS)
        self.views = int(c.VIEWS)
        self.alpha = float(c.EMA_ALPHA)
        self.restore_prob = float(c.RESTORE_PROB)
        self.w_cons = float(c.W_CONSISTENCY)
        self.w_cl = float(c.W_CONTRASTIVE)
        self.temperature = float(c.TEMPERATURE)
        self.pseudo_thr = float(c.PSEUDO_SCORE_THR)
        self.pseudo_nms = float(c.PSEUDO_NMS_IOU)
        self.levels = tuple(c.FEATURE_LEVELS)
        self.predict_mode = str(c.PREDICT)
        assert self.predict_mode in ("last_step", "post")
        self.bgr = cfg.INPUT.FORMAT == "BGR"

        self.student = student
        self.teacher = teacher
        self.source_state = {k: v.detach().clone() for k, v in student.state_dict().items()}

        decay, no_decay = [], []
        for name, p in student.named_parameters():
            (no_decay if name.endswith("bias") else decay).append(p)
        self.optimizer = torch.optim.SGD(
            [{"params": decay, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.NESTEROV,
        )
        self.optimizer_init_state = deepcopy(self.optimizer.state_dict())

        out_shape = student.backbone.output_shape()
        self.level_strides = {n: float(out_shape[n].stride) for n in self.levels}
        self.iter = 0

        log_run_provenance(
            cfg,
            method="MT+CL+SR (Moraiti et al. port)",
            extra={
                "steps": self.steps, "views": self.views, "ema_alpha": self.alpha,
                "restore_prob": self.restore_prob, "w_consistency": self.w_cons,
                "w_contrastive": self.w_cl, "temperature": self.temperature,
                "pseudo_thr": self.pseudo_thr, "levels": self.levels,
                "predict": self.predict_mode, "lr": cfg.SOLVER.BASE_LR,
                "semantic_branch_used": False,
                "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad),
            },
            logger=logger,
        )

    @classmethod
    def from_config(cls, cfg):
        def build():
            m = GeneralizedRCNN(cfg)
            DetectionCheckpointer(m).load(cfg.MODEL.WEIGHTS)
            m.to(torch.device(cfg.MODEL.DEVICE))
            m.eval()
            return m

        student = build()
        student.requires_grad_(True)
        teacher = build()
        teacher.requires_grad_(False)
        return {"student": student, "teacher": teacher, "cfg": cfg}

    @property
    def device(self):
        return self.student.pixel_mean.device

    # ------------------------------------------------------------------ #
    def _raw_forward(self, model, images: ImageList, features: Dict[str, torch.Tensor], proposals=None):
        """RPN raw outputs, proposals and ROI-head outputs on `proposals` (or on the
        model's own proposals when None)."""
        rpn = model.proposal_generator
        rpn_feats = [features[f] for f in rpn.in_features]
        obj, deltas = rpn.rpn_head(rpn_feats)
        if proposals is None:
            proposals, _ = rpn(images, features, None)
        roi = model.roi_heads
        box_in = [features[f] for f in roi.box_in_features]
        box_feats = roi.box_head(roi.box_pooler(box_in, [p.proposal_boxes for p in proposals]))
        scores, box_deltas = roi.box_predictor(box_feats)
        return obj, deltas, proposals, scores, box_deltas

    @staticmethod
    def _mse_sum(a: List[torch.Tensor], b: List[torch.Tensor]):
        total, n = 0.0, 0
        for x, y in zip(a, b):
            d = F.mse_loss(x, y, reduction="none")
            total = total + d.sum()
            n += d.numel()
        return total / max(n, 1)

    def _make_views(self, batched_inputs):
        img = batched_inputs[0]["image"].to(self.device)
        rgb = img.flip(0) if self.bgr else img
        views = []
        for _ in range(self.views):
            v = strong_view(rgb)
            v = v.flip(0) if self.bgr else v
            views.append((v.float() - self.student.pixel_mean) / self.student.pixel_std)
        return ImageList.from_tensors(
            views, self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints,
        )

    @torch.no_grad()
    def _teacher_pass(self, images_t: ImageList):
        t = self.teacher
        feats = t.backbone(images_t.tensor)
        obj, deltas, proposals, scores, box_deltas = self._raw_forward(t, images_t, feats)
        dets, _ = t.roi_heads.box_predictor.inference((scores, box_deltas), proposals)
        return feats, obj, deltas, proposals, scores, box_deltas, dets

    def _pseudo_boxes(self, dets):
        inst = dets[0]
        keep = inst.scores >= self.pseudo_thr
        boxes, labels, scores = inst.pred_boxes.tensor[keep], inst.pred_classes[keep], inst.scores[keep]
        if len(boxes):
            k = batched_nms(boxes, scores, labels, self.pseudo_nms)
            boxes, labels = boxes[k], labels[k]
        return boxes, labels

    def _object_features(self, feature_map, boxes, stride):
        f = roi_align(feature_map, [boxes], output_size=1, spatial_scale=1.0 / stride, aligned=True)
        return F.normalize(torch.flatten(f, 1), dim=1)

    def _contrastive(self, feats_t, feats_s, boxes, labels):
        if len(boxes) == 0:
            return torch.zeros((), device=self.device)
        total = 0.0
        for lvl in self.levels:
            per_view = [self._object_features(feats_t[lvl][:1], boxes, self.level_strides[lvl])]
            for v in range(self.views):
                per_view.append(self._object_features(feats_s[lvl][v:v + 1], boxes, self.level_strides[lvl]))
            total = total + supcon_teacher_anchor(torch.stack(per_view, dim=1), labels, self.temperature)
        return total

    @torch.no_grad()
    def _ema_update(self):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.mul_(self.alpha).add_(ps.detach(), alpha=1.0 - self.alpha)
        for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
            bt.copy_(bs)

    @torch.no_grad()
    def _stochastic_restore(self):
        if self.restore_prob <= 0.0:
            return
        for name, p in self.student.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                mask = torch.rand_like(p) < self.restore_prob
                p.data = torch.where(mask, self.source_state[name], p.data)

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def forward(self, batched_inputs):
        assert len(batched_inputs) == 1, "online protocol: batch size 1"
        self.iter += 1
        self.student.eval()
        self.teacher.eval()

        images_t = self.teacher.preprocess_image(batched_inputs)
        images_s = self._make_views(batched_inputs)
        assert images_s.tensor.shape[-2:] == images_t.tensor.shape[-2:]

        last_dets = None
        log = {}
        for step in range(self.steps):
            feats_t, t_obj, t_dlt, props, t_sc, t_bd, dets = self._teacher_pass(images_t)
            last_dets = dets
            boxes, labels = self._pseudo_boxes(dets)

            feats_s = self.student.backbone(images_s.tensor)
            s_obj, s_dlt, _, s_sc, s_bd = self._raw_forward(
                self.student, images_s, feats_s, proposals=[props[0]] * self.views
            )
            V = self.views
            l_rpn_obj = self._mse_sum([o.repeat_interleave(V, 0) for o in t_obj], s_obj)
            l_rpn_reg = self._mse_sum([d.repeat_interleave(V, 0) for d in t_dlt], s_dlt)
            l_roi_cls = F.mse_loss(s_sc, t_sc.repeat(V, 1))
            l_roi_reg = F.mse_loss(s_bd, t_bd.repeat(V, 1))
            consistency = l_rpn_obj + l_rpn_reg + l_roi_cls + l_roi_reg
            contrastive = self._contrastive(feats_t, feats_s, boxes, labels)
            loss = self.w_cons * consistency + self.w_cl * contrastive
            assert torch.isfinite(loss), (consistency, contrastive)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self._ema_update()
            log = {"cons": float(consistency), "cl": float(contrastive), "n_pseudo": int(len(boxes))}

        self._stochastic_restore()
        self.optimizer.load_state_dict(self.optimizer_init_state)

        if self.iter % 50 == 0:
            print(f"[MTCLSR] iter={self.iter} " + " ".join(f"{k}={v:.4g}" for k, v in log.items()))

        if self.predict_mode == "post":
            with torch.no_grad():
                _, _, _, _, _, _, last_dets = self._teacher_pass(images_t)
        results = GeneralizedRCNN._postprocess(last_dets, batched_inputs, images_t.image_sizes)
        return results
