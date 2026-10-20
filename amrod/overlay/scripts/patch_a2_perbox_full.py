#!/usr/bin/env python3
from pathlib import Path
import shutil

repo = Path.cwd()
model_path = repo / "detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py"
defaults_path = repo / "detectron2/detectron2/config/defaults.py"

if not model_path.exists():
    raise SystemExit(f"Run from AMROD repo root; missing {model_path}")

text = model_path.read_text()

old = '_valid_ctcr_modes = {"full_box", "hard_seg", "soft_seg"}'
new = '_valid_ctcr_modes = {"full_box", "per_box_full", "hard_seg", "soft_seg"}'
if new not in text:
    if old not in text:
        raise SystemExit("Could not find expected CT-CR mode validation block. Refusing to patch.")
    text = text.replace(old, new, 1)

old_block = '''        # B/C) New segmentation-supported modes.
        if teacher_seg_probs is None:
            return None

        B, K, H, W = s_seg_logits.shape
        probs = teacher_seg_probs.detach()

        if probs.shape[-2:] != (H, W):
            probs = F.interpolate(
                probs.float(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
'''
new_block = '''        # A2/B/C) Per-box CT-CR modes.
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
'''
if new_block not in text:
    if old_block not in text:
        raise SystemExit("Could not find expected B/C CT-CR setup block. Refusing to patch.")
    text = text.replace(old_block, new_block, 1)

old_guard = '''            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K or seg_c >= probs.shape[1]:
                continue
'''
new_guard = '''            seg_c = _DET_TO_SEG_CLASS_CITYSCAPES[c]
            if seg_c >= K:
                continue
            if probs is not None and seg_c >= probs.shape[1]:
                continue
'''
if new_guard not in text:
    if old_guard not in text:
        raise SystemExit("Could not find expected per-box class guard. Refusing to patch.")
    text = text.replace(old_guard, new_guard, 1)

old_loss = '''            q = probs[
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
'''
new_loss = '''            if self.ctcr_mode == "per_box_full":
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
'''
if new_loss not in text:
    if old_loss not in text:
        raise SystemExit("Could not find expected B/C loss block. Refusing to patch.")
    text = text.replace(old_loss, new_loss, 1)

backup = model_path.with_suffix(model_path.suffix + ".bak_before_a2")
if not backup.exists():
    shutil.copy2(model_path, backup)
model_path.write_text(text)

if defaults_path.exists():
    d = defaults_path.read_text()
    old_doc = '''# CT-CR spatial supervision modes.
# "full_box" preserves the legacy reproduction path exactly.
# "hard_seg" keeps only teacher-semantic-supported pixels in each bbox.
# "soft_seg" weights bbox pixels by teacher semantic probability.
'''
    new_doc = '''# CT-CR spatial supervision modes.
# "full_box" preserves the legacy reproduction path exactly.
# "per_box_full" is the A2 control: full bbox, but per-box normalized loss.
# "hard_seg" keeps only teacher-semantic-supported pixels in each bbox.
# "soft_seg" weights bbox pixels by teacher semantic probability.
'''
    if new_doc not in d and old_doc in d:
        b = defaults_path.with_suffix(defaults_path.suffix + ".bak_before_a2")
        if not b.exists():
            shutil.copy2(defaults_path, b)
        defaults_path.write_text(d.replace(old_doc, new_doc, 1))

print("A2 patch applied successfully.")
print(f"Modified: {model_path}")
print(f"Backup  : {backup}")
print("Mode    : per_box_full")
