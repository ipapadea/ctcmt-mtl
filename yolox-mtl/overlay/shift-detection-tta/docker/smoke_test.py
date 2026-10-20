"""Smoke test: verify the shift-detection-tta stack loads inside the Docker image.

Run via `docker/run.sh python docker/smoke_test.py`.
"""
from __future__ import annotations

import sys

def main() -> int:
    import torch, torchvision
    import mmengine, mmcv, mmdet, mmtrack, mmcls

    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__}"
          f"  cuda={torch.cuda.is_available()}  ngpu={torch.cuda.device_count()}")
    print(f"torchvision : {torchvision.__version__}")
    print(f"mmengine    : {mmengine.__version__}")
    print(f"mmcv        : {mmcv.__version__}")
    print(f"mmdet       : {mmdet.__version__}")
    print(f"mmtrack     : {mmtrack.__version__}")
    print(f"mmcls       : {mmcls.__version__}")

    import shift_tta
    from shift_tta.models.detectors import YOLOXMTLDetector
    from shift_tta.models.heads import YOLOXSemSegHead
    from shift_tta.models.losses import MTLConsistencyLoss, CrossTaskContrastiveLoss
    from shift_tta.models.adapters import MeanTeacherYOLOXMTLAdapterCTCMT
    print(f"shift_tta   : {shift_tta.__file__}")
    print("MTL classes imported OK")

    from mmcv.ops import RoIAlign  # noqa: F401
    print("mmcv RoIAlign import OK")

    # Instantiate the seg head at YOLOX-X sizes and run a forward pass on GPU.
    head = YOLOXSemSegHead(in_channels=(320, 320, 320), num_classes=19).cuda()
    feats = [torch.randn(1, 320, 76, 96).cuda(),
             torch.randn(1, 320, 38, 48).cuda(),
             torch.randn(1, 320, 19, 24).cuda()]
    with torch.no_grad():
        logits = head(feats)
    print(f"YOLOXSemSegHead forward OK: logits shape = {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
