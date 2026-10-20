"""CTCMT (final method) for the YOLOX-MTL detector.

Re-implementation, on top of ``YOLOXMTLDetector`` (YOLOX-X + semantic-segmentation head on the shared PAFPN),
of the method defined by ``CTCMT_MTL`` in the AMROD repository (``detectron2/modeling/meta_arch/ctcmt_mtl.py``)
with its headline configuration. One frame (batch size 1) goes through:

 1. teacher (EMA) inference on the weak view -> boxes (score >= 0.05, NMS 0.5), per-class dynamic thresholds,
    score-EMA gate;
 2. student forward on the strong view (photometric / erasing augmentation, same pixel grid);
 3. L = L_det (hard pseudo-labels, YOLOX loss, only if the gate is open and there are pseudo-boxes)
      + L_seg (soft CE vs teacher softmax, class-balanced, loss-scale preserving)
      + 0.1 * KL(anchor marginal || student marginal)
      + w_ctcr * L_CTCR (box -> semantic class, w = 0.2 + 0.8 q_T, ignore outside boxes, global normalisation);
 4. one SGD step, teacher EMA, stochastic restore towards the frozen source model;
 5. final prediction by the EMA teacher on the weak view AFTER the update (adapt-then-predict).

``adapt_enabled=False`` turns the adapter into the "Source" row: the teacher stays equal to the source model and
the very same prediction path is used. No CT-CL, no CTPV, no Fisher / gradient routing (not part of the final method).
BatchNorm layers stay in eval mode (running statistics frozen), like the FrozenBN of the Panoptic-FPN source.

Documented differences from ``CTCMT_MTL`` (Panoptic-FPN), needed because the YOLOX objectness loss produces rare
gradient spikes (norm ~4e4 vs ~3e2 typical) that otherwise drive the weights to NaN within tens of frames:
  * ``max_grad_norm``: the global gradient norm of the student is clipped to this value before the SGD step
    (``None`` = no clipping, as in ``CTCMT_MTL``);
  * non-finite guard: if the total loss or the gradient norm is not finite, the optimizer step is skipped
    (the teacher EMA and the restore still run); the number of skipped steps is logged.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from mmengine.structures import InstanceData, PixelData

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter

# detection class index (person, rider, car, truck, bus, train, motorcycle, bicycle) -> Cityscapes trainId
_DET_TO_SEG = (11, 12, 13, 14, 15, 16, 17, 18)
_SHARED_PREFIXES = ('yolox.backbone.', 'yolox.neck.')


def _dyn_thresholds(prev, per_class_mean, alpha, gamma, lo, hi):
    new = []
    for th, mean in zip(prev, per_class_mean):
        if mean > 0:
            th = gamma * th + (1.0 - gamma) * alpha * math.sqrt(mean)
        new.append(max(min(th, hi), lo))
    return new


@MODELS.register_module()
class MeanTeacherYOLOXMTLAdapterCTCMTFinal(BaseAdapter):

    def __init__(
        self,
        adapt_enabled: bool = True,
        # optimisation
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.9998,
        # losses
        weight_det: float = 1.0,
        weight_seg: float = 1.0,
        weight_ctcr: float = 0.3,
        ctcr_weight_floor: float = 0.2,
        anchor_marginal_weight: float = 0.1,
        class_balance_beta: float = 0.5,
        class_marginal_ema: float = 0.999,
        # pseudo-labels, thresholds, score-EMA gate
        pseudo_score_thr: float = 0.05,
        pseudo_nms_iou: float = 0.5,
        num_classes: int = 8,
        threshold_init: float = 0.8,
        threshold_mini: float = 0.7,
        threshold_max: float = 0.9,
        alpha_dt: float = 1.3,
        gamma_dt: float = 0.95,
        score_em_init: float = 0.5,
        score_gamma: float = 0.7,
        score_thresh: float = 1.4,
        # semantic augmentation averaging
        seg_aug_conf_thresh: float = 0.95,
        seg_aug_scales: Tuple[float, ...] = (0.75, 1.0, 1.25),
        seg_aug_flips: Tuple[bool, ...] = (False, True),
        # stochastic restore
        rst_prob: float = 0.01,
        backbone_rst_factor: float = 0.1,
        log_interval: int = 50,
        diag_grad_norms: bool = False,
        seed: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.adapt_enabled = bool(adapt_enabled)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.ema_decay = ema_decay
        self.weight_det, self.weight_seg = weight_det, weight_seg
        self.weight_ctcr, self.ctcr_weight_floor = weight_ctcr, ctcr_weight_floor
        self.anchor_marginal_weight = anchor_marginal_weight
        self.class_balance_beta, self.class_marginal_ema = class_balance_beta, class_marginal_ema
        self.pseudo_score_thr, self.pseudo_nms_iou = pseudo_score_thr, pseudo_nms_iou
        self.num_classes = num_classes
        self.thresholds = [threshold_init] * num_classes
        self.threshold_mini, self.threshold_max = threshold_mini, threshold_max
        self.alpha_dt, self.gamma_dt = alpha_dt, gamma_dt
        self.score_em = score_em_init
        self.score_gamma, self.score_thresh = score_gamma, score_thresh
        self.seg_aug_conf_thresh = seg_aug_conf_thresh
        self.seg_aug_scales, self.seg_aug_flips = tuple(seg_aug_scales), tuple(seg_aug_flips)
        self.rst_prob, self.backbone_rst_factor = rst_prob, backbone_rst_factor
        self.log_interval = log_interval
        self.diag_grad_norms = diag_grad_norms
        self.seed = seed
        self.max_grad_norm = max_grad_norm
        self._n_clipped = 0
        self._n_skipped = 0

        self.teacher = None
        self.anchor = None
        self.optimizer = None
        self.iter = 0
        self._class_marginal = None
        self._seg_aug_fired = 0
        self._grad_norm = 0.0
        self._gate_open = 0
        self._stats = dict(n_pseudo=0.0, n_det_steps=0)

        self._color_jitter = T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self._erasers = [
            T.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value='random'),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value='random'),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value='random'),
        ]

    # ------------------------------------------------------------------ setup
    def _init_source_model_state(self, model) -> None:
        if self.seed is not None:  # tools/test.py fixes seed 1234; this reseeds augmentation / restore noise
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
        det = model.detector
        det.eval()
        self.teacher = deepcopy(det).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.anchor = deepcopy(det).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self._source_params = {n: p for n, p in self.anchor.named_parameters()}
        if self.adapt_enabled:
            decay, no_decay = [], []
            for n, p in det.named_parameters():
                (no_decay if p.ndim <= 1 else decay).append(p)
            self.optimizer = torch.optim.SGD(
                [dict(params=decay, weight_decay=self.weight_decay),
                 dict(params=no_decay, weight_decay=0.0)],
                lr=self.lr, momentum=self.momentum)
        self.source_model_state = {}  # episodic reset is not used (online, no reset)

    def _adapt(self, *args, **kwargs):  # abstract in the base class
        raise NotImplementedError

    # ---------------------------------------------------------- forward helpers
    @staticmethod
    def _feats_and_outs(det, img):
        feats = det.extract_feat(img)
        return feats, det.yolox.bbox_head(feats)

    def _predict(self, det, outs, metainfo, rescale: bool, pseudo: bool) -> InstanceData:
        head = det.yolox.bbox_head
        cfg = deepcopy(head.test_cfg)
        if pseudo:
            cfg['score_thr'] = self.pseudo_score_thr
            cfg['nms'] = dict(cfg['nms'], iou_threshold=self.pseudo_nms_iou)
        return head.predict_by_feat(*outs, batch_img_metas=[metainfo], rescale=rescale, cfg=cfg)[0]

    # ------------------------------------------------------- strong augmentation
    @torch.no_grad()
    def _strong_view(self, img: torch.Tensor, metainfo) -> torch.Tensor:
        """SimCLR-style photometric + erasing augmentation (no geometry), as the AMROD ``image_strong``."""
        h, w = [int(v) for v in metainfo['img_shape'][:2]]
        x = (img[0, :, :h, :w].flip(0) / 255.0).clamp(0, 1)  # BGR -> RGB, [0, 1]
        if torch.rand(()) < 0.8:
            x = self._color_jitter(x)
        if torch.rand(()) < 0.2:
            x = TF.rgb_to_grayscale(x, num_output_channels=3)
        if torch.rand(()) < 0.5:
            sigma = float(torch.empty(()).uniform_(0.1, 2.0))
            k = 2 * int(math.ceil(3 * sigma)) + 1
            x = TF.gaussian_blur(x, [k, k], [sigma, sigma])
        for er in self._erasers:
            x = er(x)
        x = (x.clamp(0, 1) * 255.0).round().flip(0)
        out = img.clone()
        out[0, :, :h, :w] = x
        return out

    # ------------------------------------------------------ teacher pseudo-labels
    @torch.no_grad()
    def _teacher_pseudo(self, t_outs, metainfo, device):
        """Returns (boxes, labels, scores, det_gate)."""
        inst = self._predict(self.teacher, t_outs, metainfo, rescale=False, pseudo=True)
        empty = (inst.bboxes[:0], inst.labels[:0], inst.scores[:0])
        if len(inst) == 0:
            return (*empty, False)
        valid = inst.scores > 0.1
        if not valid.any():
            # no dynamic filtering on this path: the raw detector output is returned, gate closed
            return inst.bboxes, inst.labels.long(), inst.scores, False
        mean_all = float(inst.scores[valid].mean())
        ratio = mean_all / self.score_em
        gate = not (ratio > self.score_thresh or (1.0 / max(ratio, 1e-6)) > self.score_thresh)
        self.score_em = self.score_gamma * self.score_em + (1.0 - self.score_gamma) * mean_all

        per_class = [0.0] * self.num_classes
        cls, sc = inst.labels[valid], inst.scores[valid]
        for c in range(self.num_classes):
            m = cls == c
            if int(m.sum()) > 0:
                per_class[c] = float(sc[m].mean())
        self.thresholds = _dyn_thresholds(self.thresholds, per_class, self.alpha_dt, self.gamma_dt,
                                          self.threshold_mini, self.threshold_max)
        thr = torch.tensor(self.thresholds, device=device)
        keep = inst.scores >= thr[inst.labels.long()]
        return inst.bboxes[keep], inst.labels[keep].long(), inst.scores[keep], gate

    # ------------------------------------------------- teacher aug-averaged seg
    @torch.no_grad()
    def _teacher_aug_avg_probs(self, x: torch.Tensor, out_hw) -> torch.Tensor:
        H, W = x.shape[-2:]
        div = 32

        def snap(v):
            v = max(int(round(v)), div)
            return ((v + div - 1) // div) * div

        acc, n = None, 0
        for s in self.seg_aug_scales:
            Hs, Ws = snap(H * s), snap(W * s)
            for flip in self.seg_aug_flips:
                xa = F.interpolate(x, size=(Hs, Ws), mode='bilinear', align_corners=False)
                if flip:
                    xa = torch.flip(xa, dims=[-1])
                logits = self.teacher.seg_head(self.teacher.extract_feat(xa))
                p = F.interpolate(logits.float(), size=(Hs, Ws), mode='bilinear',
                                  align_corners=False).softmax(dim=1)
                if flip:
                    p = torch.flip(p, dims=[-1])
                p = F.interpolate(p, size=out_hw, mode='bilinear', align_corners=False)
                acc = p if acc is None else acc + p
                n += 1
        return acc / float(n)

    # ------------------------------------------------------------------- CT-CR
    def _ctcr_loss(self, s_logits, boxes, labels, probs) -> Optional[torch.Tensor]:
        _, K, H, W = s_logits.shape
        target = torch.full((1, H, W), 255, dtype=torch.long, device=s_logits.device)
        weight = torch.zeros((1, H, W), dtype=torch.float32, device=s_logits.device)
        sx, sy = W / self._img_w, H / self._img_h
        for (x1, y1, x2, y2), c in zip(boxes.tolist(), labels.tolist()):
            if not (0 <= c < len(_DET_TO_SEG)):
                continue
            seg_c = _DET_TO_SEG[c]
            if seg_c >= K:
                continue
            x1i, y1i = max(int(round(x1 * sx)), 0), max(int(round(y1 * sy)), 0)
            x2i, y2i = min(int(round(x2 * sx)), W), min(int(round(y2 * sy)), H)
            if x2i <= x1i or y2i <= y1i:
                continue
            target[0, y1i:y2i, x1i:x2i] = seg_c  # later boxes overwrite earlier ones
            q = probs[0, seg_c, y1i:y2i, x1i:x2i].float().clamp(0.0, 1.0)
            weight[0, y1i:y2i, x1i:x2i] = self.ctcr_weight_floor + (1.0 - self.ctcr_weight_floor) * q
        valid = target != 255
        if int(valid.sum()) == 0:
            return None
        ce = F.cross_entropy(s_logits, target, ignore_index=255, reduction='none')
        w = weight * valid.float()
        return (w * ce).sum() / w.sum().clamp_min(1e-6)

    # ------------------------------------------------------------ EMA / restore
    @torch.no_grad()
    def _update_teacher(self, det):
        d = self.ema_decay
        s_state = det.state_dict()
        for k, v in self.teacher.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s_state[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(s_state[k])

    @torch.no_grad()
    def _stochastic_restore(self, det):
        if self.rst_prob <= 0.0:
            return
        for name, p in det.named_parameters():
            if not (name.endswith('.weight') or name.endswith('.bias')) or not p.requires_grad:
                continue
            src = self._source_params.get(name)
            if src is None:
                continue
            rst = self.rst_prob
            if name.startswith(_SHARED_PREFIXES):
                rst *= self.backbone_rst_factor
            mask = (torch.rand_like(p) < rst).float()
            p.data.mul_(1.0 - mask).add_(src.data * mask)

    # -------------------------------------------------------------- prediction
    @torch.no_grad()
    def _final_prediction(self, img, data_sample):
        det = self.teacher
        feats, outs = self._feats_and_outs(det, img)
        pred = self._predict(det, outs, data_sample.metainfo, rescale=True, pseudo=False)
        logits = det.seg_head(feats)
        seg = det.seg_head.predict(logits, img.shape[-2:])
        sem = seg[0]
        data_sample.pred_sem_seg = PixelData(sem_seg=sem[None])
        return pred.clone()

    # ---------------------------------------------------------------- main step
    def adapt(self, model, img, feats, data_sample, **kwargs) -> InstanceData:
        if self.teacher is None:
            self._init_source_model_state(model)
        if not self.adapt_enabled:
            return self._final_prediction(img, data_sample)

        self.iter += 1
        det = model.detector
        det.eval()
        meta = data_sample.metainfo
        self._img_h, self._img_w = [int(v) for v in meta['img_shape'][:2]]
        device = img.device
        loss_dict: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            t_feats, t_outs = self._feats_and_outs(self.teacher, img)
            t_logits = self.teacher.seg_head(t_feats)
        boxes, labels, scores, det_gate = self._teacher_pseudo(t_outs, meta, device)
        n_pseudo = len(boxes)
        self._gate_open += int(det_gate)

        with torch.enable_grad():
            img_s = self._strong_view(img, meta)
            s_feats, s_outs = self._feats_and_outs(det, img_s)
            s_logits = det.seg_head(s_feats)

            if det_gate and n_pseudo > 0:
                gt = InstanceData(bboxes=boxes.detach(), labels=labels.detach())
                det_losses = det.yolox.bbox_head.loss_by_feat(
                    *s_outs, batch_gt_instances=[gt], batch_img_metas=[meta])
                for k, v in det_losses.items():
                    loss_dict[f'det/{k}'] = self.weight_det * v
                self._stats['n_det_steps'] += 1

            # ---- semantic soft-CE with conditional teacher aug-averaging
            q_T = t_logits.float().softmax(dim=1)
            with torch.no_grad():
                a_probs = self.anchor.seg_head(self.anchor.extract_feat(img_s)).float().softmax(dim=1)
                if float(a_probs.max(dim=1)[0].mean()) < self.seg_aug_conf_thresh:
                    q_T = self._teacher_aug_avg_probs(img_s, s_logits.shape[-2:])
                    self._seg_aug_fired += 1
                q_T = q_T.detach()
                marg = q_T.mean(dim=(0, 2, 3)).float()
                if self._class_marginal is None:
                    self._class_marginal = marg.clone()
                else:
                    self._class_marginal.mul_(self.class_marginal_ema).add_(
                        marg, alpha=1.0 - self.class_marginal_ema)
                inv = self._class_marginal.clamp_min(1e-6).pow(-self.class_balance_beta)
                cb_w = inv[q_T.argmax(dim=1)]
                cb_w = cb_w / cb_w.mean().clamp_min(1e-6)
            log_p = F.log_softmax(s_logits.float(), dim=1)
            ce = -(q_T * log_p).sum(dim=1)
            l_seg = (cb_w * ce).sum() / (cb_w.sum() + 1e-6)
            with torch.no_grad():
                scale = ce.mean() / l_seg.clamp_min(1e-8)
            loss_dict['seg/soft_ce'] = self.weight_seg * l_seg * scale

            if self.anchor_marginal_weight > 0:
                qa = a_probs.mean(dim=(0, 2, 3)).detach().clamp_min(1e-8)
                ps = log_p.exp().mean(dim=(0, 2, 3)).clamp_min(1e-8)
                loss_dict['seg/anchor_marginal'] = self.anchor_marginal_weight * (
                    qa * (qa.log() - ps.log())).sum()

            if self.weight_ctcr > 0 and n_pseudo > 0:
                l_ctcr = self._ctcr_loss(s_logits.float(), boxes, labels, q_T)
                if l_ctcr is not None:
                    loss_dict['ctcr'] = self.weight_ctcr * l_ctcr

            self.optimizer.zero_grad(set_to_none=True)
            total = sum(loss_dict.values())
            if self.diag_grad_norms and self.iter % self.log_interval == 0:
                params = [p for p in det.parameters() if p.requires_grad]
                parts = []
                for k, v in loss_dict.items():
                    g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
                    parts.append(f'{k}:{math.sqrt(sum(float(x.pow(2).sum()) for x in g if x is not None)):.2f}')
                print(f'[CTCMT-YOLOX-GRAD] iter={self.iter} ' + ' '.join(parts), flush=True)
            if torch.isfinite(total):
                total.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(
                    det.parameters(), self.max_grad_norm if self.max_grad_norm else float('inf')))
            else:
                gn = float('nan')
            self._grad_norm = gn
            if math.isfinite(gn):
                self._n_clipped += int(bool(self.max_grad_norm) and gn > self.max_grad_norm)
                self.optimizer.step()
            else:
                self._n_skipped += 1
                self.optimizer.zero_grad(set_to_none=True)

        self._update_teacher(det)
        self._stochastic_restore(det)

        if self.iter % self.log_interval == 0:
            summary = ' '.join(f'{k}={float(v.detach()):.3f}' for k, v in loss_dict.items())
            thr = self.thresholds
            print(f'[CTCMT-YOLOX] iter={self.iter} score_em={self.score_em:.3f} n_pseudo={n_pseudo} '
                  f'gate_open={self._gate_open}/{self.iter} thr={min(thr):.3f}/{sum(thr) / len(thr):.3f}/'
                  f'{max(thr):.3f} gnorm={self._grad_norm:.2f} clipped={self._n_clipped} skipped={self._n_skipped} segaug={self._seg_aug_fired}/{self.iter} {summary}', flush=True)

        return self._final_prediction(img, data_sample)
