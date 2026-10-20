"""Diagnostic: compare CoTTA_SemSeg's teacher output vs plain SemanticSegmentor
on a single ACDC fog batch to find the mIoU-collapse cause."""
import os
import torch
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.build import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader


def main() -> None:
    cfg = get_cfg()
    cfg.merge_from_file(
        "/workspace/amrod/detectron2/configs/Cityscapes/cotta_semseg_R_50_ACDC.yaml"
    )
    cfg.OUTPUT_DIR = "/tmp/probe_cotta"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    model = build_model(cfg)
    print("[PROBE post-build-model, PRE outer resume_or_load]")
    print("  teacher stem rv[:3]:", model.teacher.backbone.bottom_up.stem.conv1.norm.running_var[:3].tolist())
    print("  teacher stem rm[:3]:", model.teacher.backbone.bottom_up.stem.conv1.norm.running_mean[:3].tolist())
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    print("[PROBE post outer resume_or_load]")
    print("  teacher stem rv[:3]:", model.teacher.backbone.bottom_up.stem.conv1.norm.running_var[:3].tolist())
    print("[PROBE] model class:", type(model).__name__)
    print("[PROBE] student stem sum:", model.student.backbone.bottom_up.stem.conv1.weight.sum().item())
    print("[PROBE] teacher stem sum:", model.teacher.backbone.bottom_up.stem.conv1.weight.sum().item())
    print("[PROBE] anchor  stem sum:", model.anchor.backbone.bottom_up.stem.conv1.weight.sum().item())

    cfg2 = cfg.clone()
    cfg2.MODEL.META_ARCHITECTURE = "SemanticSegmentor"
    plain = build_model(cfg2)
    DetectionCheckpointer(plain).load(cfg2.MODEL.WEIGHTS)
    print("[PROBE] plain   stem sum:", plain.backbone.bottom_up.stem.conv1.weight.sum().item())

    # Full-model parameter fingerprint.
    def _fingerprint(m):
        return sum(float(p.detach().float().sum().item()) for p in m.parameters())

    print("[PROBE] teacher param sum:", _fingerprint(model.teacher))
    print("[PROBE] plain   param sum:", _fingerprint(plain))

    # Compare state_dict key sets and last few named params.
    t_keys = set(model.teacher.state_dict().keys())
    p_keys = set(plain.state_dict().keys())
    print("[PROBE] teacher key count:", len(t_keys), " plain key count:", len(p_keys))
    print("[PROBE] only in teacher:", list(sorted(t_keys - p_keys))[:5])
    print("[PROBE] only in plain:  ", list(sorted(p_keys - t_keys))[:5])
    # Last conv weight in the head - most likely to reveal weight-loading skew.
    last_key = "sem_seg_head.predictor.weight"
    ts = model.teacher.state_dict()[last_key].sum().item()
    ps = plain.state_dict()[last_key].sum().item()
    print(f"[PROBE] {last_key}: teacher={ts:.6f} plain={ps:.6f}")

    # Compare buffers - identical params but different outputs suggests buffers.
    def _buffer_fingerprint(m):
        s = 0.0
        n = 0
        for name, b in m.named_buffers():
            if b is not None and b.dtype.is_floating_point:
                s += float(b.detach().float().sum().item())
                n += b.numel()
        return s, n

    print("[PROBE] teacher buffer sum/count:", _buffer_fingerprint(model.teacher))
    print("[PROBE] plain   buffer sum/count:", _buffer_fingerprint(plain))
    # First few buffer keys where they diverge:
    t_bufs = dict(model.teacher.named_buffers())
    p_bufs = dict(plain.named_buffers())
    # Directly inspect FrozenBN's running_var at stem in both models.
    t_rv = model.teacher.backbone.bottom_up.stem.conv1.norm.running_var
    p_rv = plain.backbone.bottom_up.stem.conv1.norm.running_var
    print("[PROBE] teacher stem running_var[:5]:", t_rv[:5].tolist())
    print("[PROBE] plain   stem running_var[:5]:", p_rv[:5].tolist())
    print("[PROBE] teacher stem running_mean[:5]:", model.teacher.backbone.bottom_up.stem.conv1.norm.running_mean[:5].tolist())
    print("[PROBE] plain   stem running_mean[:5]:", plain.backbone.bottom_up.stem.conv1.norm.running_mean[:5].tolist())
    print("[PROBE] teacher norm class:", type(model.teacher.backbone.bottom_up.stem.conv1.norm).__name__)
    print("[PROBE] plain   norm class:", type(plain.backbone.bottom_up.stem.conv1.norm).__name__)

    loader = build_detection_test_loader(cfg2, "acdc_fog_semseg")
    batch = next(iter(loader))
    print("[PROBE] image shape:", tuple(batch[0]["image"].shape),
          "dtype:", batch[0]["image"].dtype)

    model.teacher.eval()
    plain.eval()
    with torch.no_grad():
        out_plain = plain(batch)[0]["sem_seg"]
        out_teacher = model.teacher(batch)[0]["sem_seg"]
    print("[PROBE] plain   out shape:", tuple(out_plain.shape))
    print("[PROBE] teacher out shape:", tuple(out_teacher.shape))
    print("[PROBE] max abs diff (plain vs teacher):", (out_plain - out_teacher).abs().max().item())
    print("[PROBE] plain   argmax hist:", out_plain.argmax(0).view(-1).bincount(minlength=19).tolist())
    print("[PROBE] teacher argmax hist:", out_teacher.argmax(0).view(-1).bincount(minlength=19).tolist())


if __name__ == "__main__":
    main()
