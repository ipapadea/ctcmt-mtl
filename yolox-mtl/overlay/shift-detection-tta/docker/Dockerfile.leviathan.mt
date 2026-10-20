# Thin layer on top of shift-detection-tta:cu113 for the Mean-Teacher + CL + SR adapter of
# Moraiti et al.: its student augmentation pipeline needs `albumentations` (mmdet.Albu) and
# `mmpretrain` (mmpretrain.RandomErasing), which docker/Dockerfile does not install but the
# authors' environment.yml pins (albumentations==1.0.0, mmpretrain==1.0.0rc8).
# Every package already present in the base image is frozen with a constraints file, so that
# pip cannot upgrade numpy / scipy / opencv / torch / OpenMMLab under the compiled mmcv ops.
#
#   docker build -f docker/Dockerfile.leviathan.mt -t shift-detection-tta:cu113-mt .
FROM shift-detection-tta:cu113

RUN pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/base-constraints.txt \
 && echo 'opencv-python-headless<4.9' >> /tmp/base-constraints.txt \
 && pip install -c /tmp/base-constraints.txt "albumentations==1.0.0" "mmpretrain==1.0.0rc8" \
 && python - <<'EOF'
import cv2, numpy, scipy, mmcv, mmdet, mmengine, mmpretrain, albumentations, torch
print("numpy", numpy.__version__, "scipy", scipy.__version__, "cv2", cv2.__version__, "torch", torch.__version__)
print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__, "mmengine", mmengine.__version__,
      "mmpretrain", mmpretrain.__version__, "albumentations", albumentations.__version__)
assert numpy.__version__ == "1.22.4" and scipy.__version__ == "1.7.3", "numpy/scipy changed"
assert mmcv.__version__ == "2.0.0rc4" and mmdet.__version__ == "3.0.0rc5", "OpenMMLab pins changed"
from mmcv.ops import nms  # compiled ops must still import
EOF
