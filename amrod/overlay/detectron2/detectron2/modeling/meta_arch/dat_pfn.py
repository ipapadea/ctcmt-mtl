"""DAT-BF (same-source Panoptic-FPN port) of Distribution-Aware Tuning.

Reference: Ni et al., "Distribution-Aware Continual Test-Time Adaptation for
Semantic Segmentation", arXiv:2309.13604; released code ours.py
(github.com/RochelleNi/DAT, single_gpu_our).

This is NOT an official DAT reproduction. Three layers are kept apart:
  * official DAT: pseudo-labels from an EMA teacher gated per pixel by a frozen
    source anchor; DSP/TRP tensors chosen by gradient sensitivity on pixel groups
    split by an anchor score; selected tensors accumulated over time (PAU);
    teacher EMA 0.999; fresh Adam(lr=3e-4) every frame.
  * Panoptic-FPN adaptation: semantic-only wrapper around the canonical
    Panoptic-FPN R50 source (no detection branch is instantiated), 92 eligible
    tensors, whole-tensor selection with k = max(1, int(0.001 * N)) = 1 per group
    per frame (the released int(0.001 * N) is 0 for N = 92), strict global
    element cap (10% of eligible elements, the paper's evaluated operating point).
  * boundary-free protocol: nothing is ever reset; the per-domain 100-frame
    accumulation window is removed; the cap is the only accumulation bound.

Deviations from the released code, fixed by the locked protocol:
  * rankings use current-frame gradients only (no stale gradients; gradients are
    obtained with torch.autograd.grad and never accumulated into ``.grad``);
  * the update uses the paper's Eq. 4 full-image hard-label CE (valid pixels),
    not the masked DSP/TRP losses; DSP/TRP gradients are used for ranking only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils import comm
from detectron2.utils.run_provenance import file_md5, log_run_provenance

from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["DATPFN", "DATDivergenceError", "DATTensorSelector"]

logger = logging.getLogger(__name__)


class DATDivergenceError(RuntimeError):
    """Raised (and the run aborted) only on a non-finite numerical quantity."""


# --------------------------------------------------------------------------- #
# Selection rule (pure python so that it can be unit-tested without a model)
# --------------------------------------------------------------------------- #
class DATTensorSelector:
    """Accumulating whole-tensor selection under a strict global element cap.

    One call to :meth:`consider` handles one group (DSP or TRP) on one frame:
    only the top-ranked tensor is considered, and nothing is searched beyond it.
    """

    def __init__(self, numels: Sequence[int], cap_elements: int, k: int = 1):
        if k != 1:
            raise ValueError("locked protocol: only the top-ranked tensor per group is considered")
        self.numels = [int(n) for n in numels]
        self.cap_elements = int(cap_elements)
        self.selected: List[int] = []
        self._selected_set = set()
        self.selected_elements = 0

    @property
    def selected_sorted(self) -> List[int]:
        return sorted(self._selected_set)

    def consider(self, scores: Optional[Sequence[float]]) -> Dict:
        """``scores`` is None for an empty group, else one raw score per tensor in
        registration order. Returns {winner, score, outcome}."""
        if scores is None:
            return {"winner": None, "score": None, "outcome": "empty_group"}
        winner, best = None, 0.0
        for i, s in enumerate(scores):
            if s > best:  # strict: ties go to the first tensor in registration order
                winner, best = i, float(s)
        if winner is None:
            return {"winner": None, "score": 0.0, "outcome": "zero_score"}
        if winner in self._selected_set:
            return {"winner": winner, "score": best, "outcome": "already_selected"}
        if self.selected_elements + self.numels[winner] > self.cap_elements:
            return {"winner": winner, "score": best, "outcome": "does_not_fit"}
        self.selected.append(winner)
        self._selected_set.add(winner)
        self.selected_elements += self.numels[winner]
        return {"winner": winner, "score": best, "outcome": "added"}


# --------------------------------------------------------------------------- #
# Multi-view helpers (same scale-major / flip-minor ordering as the CoTTA/OBAO ports)
# --------------------------------------------------------------------------- #
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
                y = x if tuple(size) == (height, width) else F.interpolate(
                    x, size=size, mode="bilinear", align_corners=False
                )
                return torch.flip(y, dims=[-1]) if flip else y

            def invert(probs, size, flip=flip):
                if flip:
                    probs = torch.flip(probs, dims=[-1])
                if tuple(probs.shape[-2:]) == tuple(size):
                    return probs
                return F.interpolate(probs, size=size, mode="bilinear", align_corners=False)

            ops.append((float(scale), bool(flip), apply, invert))
    return ops


@META_ARCH_REGISTRY.register()
class DATPFN(nn.Module):
    """Semantic-only DAT-BF wrapper around a detectron2 SemanticSegmentor."""

    @configurable
    def __init__(self, *, student, teacher, anchor, hp: Dict, cfg):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.anchor = anchor
        self.hp = dict(hp)
        self.iter = 0
        self.num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        self.ignore_value = int(cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE)

        head = self.student.sem_seg_head
        if not all(hasattr(head, n) for n in ("in_features", "scale_heads", "predictor", "layers")):
            raise TypeError("DATPFN requires a SemSegFPNHead-compatible semantic head")
        if comm.get_world_size() != 1:
            raise RuntimeError("DATPFN keeps persistent online state; run on a single process")

        divisor = int(getattr(self.student.backbone, "size_divisibility", 32) or 32)
        self._aug_ops = _build_aug_ops(self.hp["aug_scales"], self.hp["aug_flips"], divisor)
        self._score_ops = _build_aug_ops(self.hp["score_scales"], (False,), divisor)
        if tuple(self.hp["score_scales"]) != (1.0, 1.25, 1.5):
            raise ValueError("locked protocol: anchor score views are scales 1.0, 1.25, 1.5 (unflipped)")
        if not any(s == 1.0 and not f for s, f, _, _ in self._aug_ops):
            raise ValueError("teacher views must contain the unflipped 1.0-scale view")

        # Eligible set: every semantic-model tensor (FREEZE_AT is overridden).
        named = list(self.student.named_parameters())
        self.param_names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.numels = [p.numel() for p in self.params]
        self._anchor_list = list(self.anchor.parameters())
        self.eligible_tensors = len(self.params)
        self.eligible_elements = int(sum(self.numels))
        if any(not p.requires_grad for p in self.params):
            raise AssertionError("all student tensors must be eligible (requires_grad=True)")
        if self.eligible_tensors != self.hp["expected_tensors"] or \
                self.eligible_elements != self.hp["expected_elements"]:
            raise AssertionError(
                f"eligible set {self.eligible_tensors} tensors / {self.eligible_elements} elements "
                f"!= locked {self.hp['expected_tensors']} / {self.hp['expected_elements']}"
            )
        if any(p.requires_grad for p in self.teacher.parameters()):
            raise AssertionError("DAT EMA teacher must be frozen")
        if any(p.requires_grad for p in self.anchor.parameters()):
            raise AssertionError("DAT source anchor must be frozen")
        if [n for n, _ in self.teacher.named_parameters()] != self.param_names or \
                [n for n, _ in self.anchor.named_parameters()] != self.param_names:
            raise AssertionError("student/teacher/anchor parameter registration order differs")
        det_keys = [n for n, _ in self.student.named_modules()
                    if n.startswith(("roi_heads", "proposal_generator"))]
        if det_keys:
            raise AssertionError(f"detection modules present in the semantic wrapper: {det_keys[:3]}")

        self.cap_elements = int(math.floor(self.hp["budget_fraction"] * self.eligible_elements))
        k = max(1, int(self.hp["k_fraction"] * self.eligible_tensors))
        self.selector = DATTensorSelector(self.numels, self.cap_elements, k=k)
        self.k = k

        self._frames_since_add = 0
        self._prev_sign: Dict[int, torch.Tensor] = {}
        self._out_dir = cfg.OUTPUT_DIR
        os.makedirs(self._out_dir, exist_ok=True)
        self._frame_log_path = os.path.join(self._out_dir, "dat_bf_frames.jsonl")
        if os.path.exists(self._frame_log_path) and os.path.getsize(self._frame_log_path) > 0:
            raise FileExistsError(f"refusing to overwrite {self._frame_log_path}")
        self._frame_log = open(self._frame_log_path, "w")
        self._save_predictions = bool(self.hp["save_predictions"])
        self._pred_dirs = {}
        if self._save_predictions:
            for tag in ("primary", "plain_view"):
                d = os.path.join(self._out_dir, "dat_bf_predictions", tag)
                os.makedirs(d, exist_ok=True)
                self._pred_dirs[tag] = d

        self._log_startup(cfg)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg):
        def build_source(frozen: bool):
            model = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            model.to(torch.device(cfg.MODEL.DEVICE))
            model.eval()
            model.requires_grad_(not frozen)  # re-enables stem/res2 (FREEZE_AT) for the student
            return model

        expected_md5 = cfg.SOLVER.DAT_EXPECTED_SOURCE_MD5
        if expected_md5:
            actual = file_md5(cfg.MODEL.WEIGHTS)
            if actual != expected_md5:
                raise AssertionError(f"source checkpoint MD5 {actual} != locked {expected_md5}")

        hp = {
            "lr": float(cfg.SOLVER.DAT_LR),
            "betas": tuple(float(b) for b in cfg.SOLVER.DAT_BETAS),
            "eps": float(cfg.SOLVER.DAT_EPS),
            "weight_decay": float(cfg.SOLVER.DAT_WEIGHT_DECAY),
            "ema_decay": float(cfg.SOLVER.DAT_EMA_DECAY),
            "anchor_gate": float(cfg.SOLVER.DAT_ANCHOR_GATE),
            "dsp_threshold": float(cfg.SOLVER.DAT_DSP_THRESHOLD),
            "trp_threshold": float(cfg.SOLVER.DAT_TRP_THRESHOLD),
            "score_scales": tuple(float(s) for s in cfg.SOLVER.DAT_SCORE_SCALES),
            "aug_scales": tuple(float(s) for s in cfg.SOLVER.DAT_AUG_SCALES),
            "aug_flips": tuple(bool(f) for f in cfg.SOLVER.DAT_AUG_FLIPS),
            "budget_fraction": float(cfg.SOLVER.DAT_BUDGET_FRACTION),
            "k_fraction": float(cfg.SOLVER.DAT_K_FRACTION),
            "expected_tensors": int(cfg.SOLVER.DAT_EXPECTED_TENSORS),
            "expected_elements": int(cfg.SOLVER.DAT_EXPECTED_ELEMENTS),
            "save_predictions": bool(cfg.SOLVER.DAT_SAVE_PREDICTIONS),
            "audit": bool(cfg.SOLVER.DAT_AUDIT),
            "official_repo_sha": str(cfg.SOLVER.DAT_OFFICIAL_REPO_SHA),
        }
        return {
            "student": build_source(frozen=False),
            "teacher": build_source(frozen=True),
            "anchor": build_source(frozen=True),
            "hp": hp,
            "cfg": cfg,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        for m in (self.student, self.teacher, self.anchor):
            m.eval()
        return self

    @property
    def device(self):
        return self.student.pixel_mean.device

    def _log_startup(self, cfg):
        extra = {
            "dat_variant": "boundary-free (DAT-BF, same-source Panoptic-FPN port)",
            "dat_official_repo_sha": self.hp["official_repo_sha"],
            "semantic_branch_only": True,
            "detection_branch_used": False,
            "detection_modules_instantiated": 0,
            "optimizer": "Adam (fresh every frame, one step)",
            "optimizer_lr": self.hp["lr"],
            "optimizer_betas": self.hp["betas"],
            "optimizer_eps": self.hp["eps"],
            "optimizer_weight_decay": self.hp["weight_decay"],
            "ema_decay": self.hp["ema_decay"],
            "eligible_tensors": self.eligible_tensors,
            "eligible_elements": self.eligible_elements,
            "selection_budget_fraction": self.hp["budget_fraction"],
            "selection_cap_elements": self.cap_elements,
            "selection_k_per_group_per_frame": self.k,
            "selected_tensors_at_start": 0,
            "selected_elements_at_start": 0,
            "selected_fraction_at_start": 0.0,
            "anchor_gate": self.hp["anchor_gate"],
            "dsp_threshold(<)": self.hp["dsp_threshold"],
            "trp_threshold(>)": self.hp["trp_threshold"],
            "anchor_score_scales": self.hp["score_scales"],
            "teacher_aug_views": len(self._aug_ops),
            "reset_occurred": False,
            "frame_log": self._frame_log_path,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "env_NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
            "env_CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
        log_run_provenance(cfg, method="DAT-BF (same-source Panoptic-FPN port)", extra=extra, logger=logger)

    # ------------------------------------------------------------------ #
    # forward passes
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logits(model, x):
        return model.sem_seg_head.layers(model.backbone(x))

    def _preprocess(self, batched_inputs):
        normalized = [
            (s["image"].to(self.device).float() - self.student.pixel_mean) / self.student.pixel_std
            for s in batched_inputs
        ]
        return ImageList.from_tensors(
            normalized,
            self.student.backbone.size_divisibility,
            padding_constraints=self.student.backbone.padding_constraints,
        )

    def _view_probs(self, model, x, ops):
        size = x.shape[-2:]
        for scale, flip, apply, invert in ops:
            xa = apply(x)
            logits = F.interpolate(self._logits(model, xa), size=xa.shape[-2:],
                                   mode="bilinear", align_corners=False)
            yield scale, flip, invert(logits.float().softmax(dim=1), size)

    @torch.no_grad()
    def _teacher_probs(self, x):
        acc, plain, n = None, None, 0
        for scale, flip, probs in self._view_probs(self.teacher, x, self._aug_ops):
            acc = probs if acc is None else acc + probs
            n += 1
            if scale == 1.0 and not flip:
                plain = probs
        return acc / n, plain

    @torch.no_grad()
    def _anchor_maps(self, x):
        """Per-view max-softmax maps of the frozen anchor at the score scales."""
        return [p.max(dim=1).values for _, _, p in self._view_probs(self.anchor, x, self._score_ops)]

    # ------------------------------------------------------------------ #
    def _abort(self, quantity: str):
        info = {"frame": self.iter, "quantity": quantity}
        with open(os.path.join(self._out_dir, "dat_bf_DIVERGED.json"), "w") as f:
            json.dump(info, f)
        self._frame_log.flush()
        raise DATDivergenceError(f"non-finite {quantity} at frame {self.iter}; run aborted (no recovery)")

    def _check_finite(self, tensor, quantity: str):
        if not bool(torch.isfinite(tensor).all()):
            self._abort(quantity)

    def _format_results(self, labels, batched_inputs, image_sizes):
        scores = F.one_hot(labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        out = []
        for score, sample, image_size in zip(scores, batched_inputs, image_sizes):
            height = sample.get("height", image_size[0])
            width = sample.get("width", image_size[1])
            out.append({"sem_seg": sem_seg_postprocess(score, image_size, height, width)})
        return out

    def _group_scores(self, loss):
        grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        scores = []
        for g in grads:
            if g is None:
                scores.append(0.0)
                continue
            self._check_finite(g, "ranking gradient")
            scores.append(float(g.abs().sum(dtype=torch.float64)))
        return grads, scores

    def _save_pred(self, tag, result, frame):
        from PIL import Image

        label = result["sem_seg"].argmax(dim=0).to(torch.uint8).cpu().numpy()
        Image.fromarray(label).save(os.path.join(self._pred_dirs[tag], f"{frame:06d}.png"))

    # ------------------------------------------------------------------ #
    @torch.enable_grad()
    def forward(self, batched_inputs):
        if len(batched_inputs) != 1:
            raise ValueError("strict online DAT-BF requires batch size 1")
        t_start = time.time()
        self.iter += 1
        self.student.eval(); self.teacher.eval(); self.anchor.eval()
        if any(p.grad is not None for p in self.params):
            raise AssertionError("stale gradient present at frame start")

        images = self._preprocess(batched_inputs)
        x = images.tensor
        height, width = x.shape[-2:]
        valid_h, valid_w = images.image_sizes[0]
        valid = torch.zeros((1, height, width), dtype=torch.bool, device=x.device)
        valid[:, :valid_h, :valid_w] = True
        n_valid = float(valid.sum())

        # ---- pre-update teacher / anchor quantities -------------------- #
        with torch.no_grad():
            aug_probs, plain_probs = self._teacher_probs(x)
            self._check_finite(aug_probs, "teacher aggregated probabilities")
            self._check_finite(plain_probs, "teacher plain-view probabilities")
            maps = self._anchor_maps(x)
            score = torch.stack(maps, dim=0).mean(dim=0)
            self._check_finite(score, "anchor score")
            gate = maps[0] > self.hp["anchor_gate"]
            plain_labels = plain_probs.argmax(dim=1)
            pseudo = torch.where(gate, plain_labels, aug_probs.argmax(dim=1))
            dsp_mask = valid & (score < self.hp["dsp_threshold"])
            trp_mask = valid & (score > self.hp["trp_threshold"])
            primary = self._format_results(pseudo, batched_inputs, images.image_sizes)
            secondary = self._format_results(plain_labels, batched_inputs, images.image_sizes)

        # ---- student forward and losses (plain scale-1.0 view, padded size) ---- #
        s_logits = F.interpolate(self._logits(self.student, x), size=(height, width),
                                 mode="bilinear", align_corners=False).float()
        ce = F.cross_entropy(s_logits, pseudo, reduction="none")  # [1, H, W]
        loss_full = (ce * valid.float()).sum() / n_valid  # Eq. 4 over valid pixels
        self._check_finite(loss_full, "loss")

        grads_group, loss_group = {}, {}
        outcomes = {}
        for name, mask in (("dsp", dsp_mask), ("trp", trp_mask)):  # DSP before TRP
            if bool(mask.any()):
                loss_g = (ce * mask.float()).sum() / n_valid
                self._check_finite(loss_g, "loss")
                grads_group[name], scores = self._group_scores(loss_g)
                loss_group[name] = float(loss_g.detach())
                res = self.selector.consider(scores)
            else:
                grads_group[name] = None
                res = self.selector.consider(None)
            res["winner_name"] = None if res["winner"] is None else self.param_names[res["winner"]]
            outcomes[name] = res

        added = [outcomes[g]["winner_name"] for g in ("dsp", "trp") if outcomes[g]["outcome"] == "added"]
        self._frames_since_add = 0 if added else self._frames_since_add + 1

        # ---- update: Eq. 4 gradient on the selected whole tensors ------------ #
        sel_idx = self.selector.selected_sorted
        step_taken = False
        diag = {}
        audit_pre = [p.detach().clone() for p in self.params] if self.hp["audit"] else None  # diagnostic only
        with torch.no_grad():
            s_pred = s_logits.argmax(dim=1)
            student_disagree = float(((s_pred != pseudo) & valid).sum()) / n_valid
            plain_vs_primary = float(((plain_labels != pseudo) & valid).sum()) / n_valid
        if sel_idx:
            sel_params = [self.params[i] for i in sel_idx]
            g_upd = torch.autograd.grad(loss_full, sel_params, allow_unused=False)
            for g in g_upd:
                self._check_finite(g, "update gradient")
            diag = self._update_diagnostics(sel_idx, g_upd, grads_group)
            before = [p.detach().clone() for p in sel_params]
            for p, g in zip(sel_params, g_upd):
                p.grad = g.detach()
            optimizer = torch.optim.Adam(
                sel_params, lr=self.hp["lr"], betas=self.hp["betas"], eps=self.hp["eps"],
                weight_decay=self.hp["weight_decay"], amsgrad=False,
            )
            optimizer.step()
            for p in sel_params:
                p.grad = None
            del optimizer
            step_taken = True
            with torch.no_grad():
                for p in sel_params:
                    self._check_finite(p, "selected parameters after step")
                delta = torch.cat([(p.detach() - b).flatten() for p, b in zip(sel_params, before)])
                base = torch.cat([b.flatten() for b in before])
                diag["delta_l2_rel"] = float(delta.norm() / base.norm().clamp_min(1e-12))
                diag["delta_max_abs"] = float(delta.abs().max())
                drift = torch.cat([(p.detach() - self._anchor_list[i].detach()).flatten()
                                   for p, i in zip(sel_params, sel_idx)])
                diag["drift_from_source_max_abs"] = float(drift.abs().max())
                diag["drift_from_source_l2"] = float(drift.norm())
            del before, g_upd

        # ---- EMA teacher after the student step (all parameters, fixed alpha) -- #
        with torch.no_grad():
            teacher_pre = [p.detach().clone() for p in self.teacher.parameters()] if self.hp["audit"] else None
            d = self.hp["ema_decay"]
            for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
                tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
        if self.hp["audit"]:
            diag.update(self._audit(audit_pre, teacher_pre, sel_idx))
            del audit_pre, teacher_pre

        # ---- logging --------------------------------------------------------- #
        frac = lambda m: float(m.sum()) / n_valid
        valid_dsp, valid_trp = frac(dsp_mask), frac(trp_mask)
        record = {
            "frame": self.iter,
            "file_name": batched_inputs[0].get("file_name"),  # record only; never read by the algorithm
            "padded_hw": [height, width], "valid_hw": [int(valid_h), int(valid_w)],
            "gate_frac": frac(gate & valid),
            "frac_dsp": valid_dsp, "frac_trp": valid_trp,
            "frac_mid": 1.0 - valid_dsp - valid_trp,
            "score_mean": float(score[valid].mean()),
            "loss_full": float(loss_full.detach()),
            "loss_dsp": loss_group.get("dsp"), "loss_trp": loss_group.get("trp"),
            "dsp": outcomes["dsp"], "trp": outcomes["trp"],
            "selected_tensors": len(sel_idx),
            "selected_elements": self.selector.selected_elements,
            "selected_fraction": self.selector.selected_elements / self.eligible_elements,
            "frames_since_last_addition": self._frames_since_add,
            "added": added,
            "step_taken": step_taken,
            "student_vs_pseudo_disagree": student_disagree,
            "plain_vs_primary_disagree": plain_vs_primary,
            "reset_occurred": False,
            "frame_ms": 1000.0 * (time.time() - t_start),
        }
        record.update(diag)
        self._frame_log.write(json.dumps(record) + "\n")
        self._frame_log.flush()
        if self.iter <= 5 or self.iter % 50 == 0 or added:
            logger.info(
                "DAT-BF iter=%d loss=%.5f dsp=%.3f trp=%.3f | DSP:%s %s | TRP:%s %s | selected=%d tensors %.4f",
                self.iter, record["loss_full"], valid_dsp, valid_trp,
                outcomes["dsp"]["outcome"], outcomes["dsp"]["winner_name"],
                outcomes["trp"]["outcome"], outcomes["trp"]["winner_name"],
                len(sel_idx), record["selected_fraction"],
            )

        if self._save_predictions:
            self._save_pred("primary", primary[0], self.iter)
            self._save_pred("plain_view", secondary[0], self.iter)
        return primary

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _audit(self, pre_student, pre_teacher, sel_idx):
        """Diagnostic-only structural audit (never aborts, never modifies state or RNG).

        * only tensors in the selected set may change on the student;
        * the teacher update must satisfy t' = d*t + (1-d)*s up to float32 rounding;
        * non-selected tensors must stay at the source values (student exactly, teacher up to
          EMA rounding).
        """
        sel = set(sel_idx)
        d = self.hp["ema_decay"]
        eps32 = 2.0 ** -23
        ema_tol = 2.0 ** -21  # ~4 float32 ulps of the operand magnitudes
        changed_nonsel, changed_nonsel_max = 0, 0.0
        unchanged_sel = 0
        nonsel_src_changed, nonsel_src_max = 0, 0.0
        ema_res_max, ema_ratio_max = 0.0, 0.0
        t_dev_max, t_dev_ratio_max = 0.0, 0.0
        teacher_params = list(self.teacher.parameters())
        for i, (p, pre) in enumerate(zip(self.params, pre_student)):
            p = p.detach()
            same = torch.equal(p, pre)
            if i in sel:
                unchanged_sel += int(same)
            elif not same:
                changed_nonsel += 1
                changed_nonsel_max = max(changed_nonsel_max, float((p - pre).abs().max()))
            src = self._anchor_list[i].detach()
            if i not in sel and not torch.equal(p, src):
                nonsel_src_changed += 1
                nonsel_src_max = max(nonsel_src_max, float((p - src).abs().max()))
            s64, tpre64, tpost = p.double(), pre_teacher[i].double(), teacher_params[i].detach().double()
            res = (tpost - (d * tpre64 + (1.0 - d) * s64)).abs()
            ema_res_max = max(ema_res_max, float(res.max()))
            ema_ratio_max = max(ema_ratio_max, float((res / (ema_tol * (tpre64.abs() + s64.abs()) + 1e-30)).max()))
            if i not in sel:
                dev = (tpost - src.double()).abs()
                t_dev_max = max(t_dev_max, float(dev.max()))
                t_dev_ratio_max = max(t_dev_ratio_max, float((dev / (eps32 * src.double().abs() + 1e-30)).max()))
        return {
            "audit_changed_nonselected_student_tensors": changed_nonsel,
            "audit_changed_nonselected_student_max_abs": changed_nonsel_max,
            "audit_unchanged_selected_student_tensors": unchanged_sel,
            "audit_nonselected_student_vs_source_changed_tensors": nonsel_src_changed,
            "audit_nonselected_student_vs_source_max_abs": nonsel_src_max,
            "audit_ema_residual_max_abs": ema_res_max,
            "audit_ema_residual_ratio_max": ema_ratio_max,  # <= 1 means within tolerance
            "audit_ema_consistent": ema_ratio_max <= 1.0,
            "audit_nonselected_teacher_dev_max_abs": t_dev_max,
            "audit_nonselected_teacher_dev_ratio_max": t_dev_ratio_max,  # in units of 2^-23 * |source|
        }

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_diagnostics(self, sel_idx, g_upd, grads_group):
        """Label-free per-frame gradient diagnostics on the selected set."""
        flat_upd = torch.cat([g.flatten() for g in g_upd])
        out = {
            "g_upd_l2": float(flat_upd.norm()),
            "g_upd_eps_frac": float((flat_upd.abs() < self.hp["eps"]).float().mean()),
        }
        flat = {"upd": flat_upd}
        for name in ("dsp", "trp"):
            grads = grads_group.get(name)
            if grads is None:
                out[f"g_{name}_l2_on_selected"] = None
                continue
            vec = torch.cat([(grads[i] if grads[i] is not None else torch.zeros_like(self.params[i])).flatten()
                             for i in sel_idx])
            flat[name] = vec
            out[f"g_{name}_l2_on_selected"] = float(vec.norm())
            both = (vec != 0) & (flat_upd != 0)
            out[f"sign_agree_upd_{name}"] = (
                float((torch.sign(vec[both]) == torch.sign(flat_upd[both])).float().mean()) if both.any() else None
            )
        # sign persistence of the update gradient on tensors selected in consecutive frames
        agree, total = 0, 0
        new_prev = {}
        for i, g in zip(sel_idx, g_upd):
            s = torch.sign(g).to(torch.int8)
            prev = self._prev_sign.get(i)
            if prev is not None:
                nz = (s != 0) & (prev != 0)
                agree += int(((s == prev) & nz).sum())
                total += int(nz.sum())
            new_prev[i] = s
        self._prev_sign = new_prev
        out["sign_persistence"] = (agree / total) if total else None
        return out
