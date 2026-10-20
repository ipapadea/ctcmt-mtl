"""Isolate: what does DetectionCheckpointer(...).load() do vs .resume_or_load()?"""
import torch
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer


def stem_rv(m):
    return m.backbone.bottom_up.stem.conv1.norm.running_var[:5].tolist()


def stem_rm(m):
    return m.backbone.bottom_up.stem.conv1.norm.running_mean[:5].tolist()


def main() -> None:
    cfg = get_cfg()
    cfg.merge_from_file("/workspace/amrod/detectron2/configs/Cityscapes/cotta_semseg_R_50_ACDC.yaml")
    weights = cfg.MODEL.WEIGHTS
    print("[STEP] weights:", weights)

    # A: plain call
    A = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
    print("[A pre-load] rv:", stem_rv(A), "rm:", stem_rm(A))
    DetectionCheckpointer(A).load(weights)
    print("[A post-load] rv:", stem_rv(A), "rm:", stem_rm(A))

    # B: same, but call save_dir specified
    B = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
    print("[B pre-load] rv:", stem_rv(B))
    DetectionCheckpointer(B, save_dir="/tmp/probe_ckpt").load(weights)
    print("[B post-load] rv:", stem_rv(B))

    # C: resume_or_load path
    C = META_ARCH_REGISTRY.get("SemanticSegmentor")(cfg)
    DetectionCheckpointer(C, save_dir="/tmp/probe_ckpt2").resume_or_load(weights, resume=True)
    print("[C post-resume_or_load] rv:", stem_rv(C))


if __name__ == "__main__":
    main()
