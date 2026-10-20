# Third-party code and data

This repository contains **our own files** and **patches** against two upstream repositories; it does not redistribute their code.
`amrod/setup.sh` and `yolox-mtl/setup.sh` fetch the upstream sources at pinned commits.

| Upstream | Pinned commit | License of the upstream |
|---|---|---|
| [ShileiCao/AMROD](https://github.com/ShileiCao/AMROD) (Detectron2 fork with AMROD, "Continual Test-Time Adaptation for Object Detection with Adaptive Monitoring and Randomized Restoration") | `199133286d60` | no license file in the repository (it vendors Detectron2, Apache-2.0, `detectron2/LICENSE`) |
| [panagiotamoraiti/shift-tta](https://github.com/panagiotamoraiti/shift-tta) (`shift-detection-tta`; MT + CL + SR of Moraiti et al., built on the SHIFT code base) | `a2449e42fc3d` | MIT, Copyright (c) 2023 ETH VIS Group (`shift-detection-tta/LICENSE`) |

Same-source ports of other methods, in `amrod/overlay/detectron2/detectron2/modeling/meta_arch/`. They derive from the official implementations named in the header of each file
and have to be used under the licenses of those implementations:

- `cotta_semseg.py`: CoTTA, Wang et al., CVPR 2022 (https://github.com/qinenergy/cotta)
- `tent_semseg.py`: TENT, Wang et al., ICLR 2021
- `obao_semseg.py`: OBAO, ECCV 2024
- `ctaod_yoo.py`: CTAOD, Yoo et al., CVPR 2024 (https://github.com/natureyoo/ContinualTTA_ObjectDetection)
- `dat_pfn.py`: DAT, Ni et al., arXiv:2309.13604 (https://github.com/RochelleNi/DAT)
- `mtclsr.py`: our port of the single-task method of Moraiti et al. (EJAI 2025) to Panoptic-FPN, written from the paper and the official `shift-tta` code.

Not included: datasets (ACDC, Cityscapes, Cityscapes-C, SHIFT: obtain them from their owners under their terms), model checkpoints (our source models, and the YOLOX-X Cityscapes checkpoint of Moraiti et al.).
