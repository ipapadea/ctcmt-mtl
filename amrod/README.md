# amrod/: CTCMT on Panoptic-FPN (Detectron2)

The reference implementation of CTCMT (`CTCMT_MTL`) and the experiments of the Panoptic-FPN R50 source model.
It is built on the AMROD Detectron2 code base of Cao et al. ([ShileiCao/AMROD](https://github.com/ShileiCao/AMROD)).
That repository has **no license file**, so it is not redistributed here: `setup.sh` fetches it at the pinned commit
`1991332` and then applies our changes (a patch for six upstream files plus our own new files).

```bash
bash setup.sh ~/amrod_ctcmt        # = ShileiCao/AMROD@1991332 + patches/amrod_upstream_changes.patch + overlay/
```

## What is added

| Path (inside the built tree) | Content |
|---|---|
| `detectron2/detectron2/modeling/meta_arch/ctcmt_mtl.py` | **CTCMT_MTL**, the method (see [../docs/canonical_ctcmt.md](../docs/canonical_ctcmt.md)); SOLVER flags `CTCMT_*` |
| `.../meta_arch/ctcmt_seg.py` | `CTCMT_Seg`: the method applied to single-task semantic segmentation (Semantic-FPN source), for the specialist-source study |
| `.../meta_arch/mtclsr.py` | **MTCLSR**: port of the single-task method of Moraiti et al. (mean teacher + contrastive learning + stochastic restoration) to the Panoptic-FPN detection branch |
| `.../meta_arch/{cotta,tent,obao}_semseg.py`, `ctaod_yoo.py`, `dat_pfn.py` | same-source ports of the baselines CoTTA, TENT, OBAO, CTAOD and DAT-BF. They derive from the official implementations referenced in the header of each file; see [../THIRD_PARTY.md](../THIRD_PARTY.md) |
| `detectron2/configs/Cityscapes/*.yaml` | configs of all experiments. Headline: `newsrc_acdc_ctcr.yaml`, `newsrc_csc_ctcr.yaml`; Base (no CT-CR): `newsrc_{acdc,csc}_base.yaml`; source model: `panoptic_fpn_R_50_segw1.yaml`; MTCLSR: `mtclsr_pfn_R_50_{ACDC,CS_C}.yaml` |
| `scripts/` | runners and batch scripts (`paper26_batch_*.sh`, `run_ctcr_acdc_local.sh`, `run_mixed_lt_local.sh`, `train_source_pfn_local.sh`, ...). They are the exact scripts that produced the results |
| `tools/` | corruption generator (`corrgen_amrod.py`), result harvesting and tabulation |
| `Dockerfile`, `Dockerfile.cu118`, `Dockerfile.train` | environments (CUDA 12.1 / CUDA 11.8 for machines with an old driver / + shapely for source training) |
| `patches/amrod_upstream_changes.patch` | our changes to `defaults.py` (all `SOLVER.CTCMT_*`, `TEST.MTCLSR` options), `builtin.py` (dataset registration), `meta_arch/__init__.py`, `amrod.py`, `semantic_seg.py` (strong-view option), `tools/train_net.py` |

`ctcmt_mtl.py` also contains the explored-but-unused machinery (CT-CL, CTPV, gradient routing, Task-CoCo, Fisher variants). All of it is flag-gated and off in the headline configs.

## Environment

The scripts run inside docker, with the repo, the datasets and the output directory bind-mounted. Build one of the images from the built tree:

```bash
cd ~/amrod_ctcmt
docker build -f Dockerfile -t amrod:latest .                 # CUDA 12.1, torch 2.1.2
docker build -f Dockerfile.cu118 -t amrod:cu118 .            # drivers <= 470: CUDA 11.8 runtime, forward-compat libs removed
docker build -f Dockerfile.train -t amrod-train:latest .     # only needed to train the source model
```

## Data and source model

- **ACDC** (`rgb_anon`, `gt`, `gt_detection`), **Cityscapes** (`leftImg8bit`, `gtFine`, COCO-format instance json) and **Cityscapes-C**: severity-5 corruptions of the Cityscapes val set, generated with `tools/corrgen_amrod.py`
  (`imagecorruptions==1.1.2`). The dataset registration is in `detectron2/detectron2/data/datasets/builtin.py`.
- Source model: `bash scripts/train_source_pfn_local.sh` (config `panoptic_fpn_R_50_segw1.yaml`, 60k iterations, semantic loss weight 1). Its checkpoint `panoptic_fpn_R50_cityscapes_segw1/model_final.pth` is **not** included.
- Streams: ACDC4 = fog, night, rain, snow (train split); CSC12 = 12 corruptions (defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression);
  ACDC-LT = ACDC4 x10; CSC-LT = fog, motion_blur, snow, brightness, defocus_blur x10.

## Running

All scripts take their locations from environment variables (`HOST_REPO`, `HOST_OUT`, `ACDC_ROOT`, `CITYSCAPES_ROOT`, `CITYSCAPES_ANN_ROOT`, `DOCKER_IMAGE`);
the defaults are the layout of the authors' workstation (`/media/ilias/DATA/ilias/...`), so set them. Examples:

```bash
# one run of the headline method (CTCMT) or of Base on ACDC4
bash scripts/run_ctcr_acdc_local.sh 0 my_ctcmt_acdc4 detectron2/configs/Cityscapes/newsrc_acdc_ctcr.yaml 0
bash scripts/run_ctcr_acdc_local.sh 0 my_base_acdc4  detectron2/configs/Cityscapes/newsrc_acdc_base.yaml 0
# Cityscapes-C
bash scripts/run_mixed_lt_local.sh ...      # see the header of the script

# predict-before-update control (CTCMT and Base, 3 seeds)         -> results/preupdate_prediction
bash scripts/paper26_batch_preupdate.sh 0
# MTCLSR port of the Moraiti et al. method                       -> results/mtclsr
bash scripts/paper26_batch_mtclsr.sh 0
```

The batch scripts refuse to overwrite existing runs (`SKIP_COMPLETE=1` / `FORCE=1` change that) and hold one lock per GPU. Many scripts under `scripts/` are exploratory (ablations, routing, retention studies);
`remote_*_leviathan.sh` and `run_remote_conda.sh` target a machine without docker and need `KEY`, `REMOTE` and `RBASE` set.

## Results in this directory

- [`results/preupdate_prediction/`](results/preupdate_prediction/summary.md): CTCMT and Base, ACDC4 and CSC12, 3 seeds, with the teacher prediction taken after the update (reported protocol) and before it (the protocol of AMROD, CTAOD, OBAO, DAT-BF).
  The difference is at most 0.15 AP50 / mIoU with varying sign; CTCMT - Base holds under both.
  The flag `SOLVER.CTCMT_PREDICT_BEFORE_UPDATE` was added by commit `e98af1b` of our research repository (it was uncommitted when the runs started, see `meta/`).
- [`results/mtclsr/`](results/mtclsr/summary.md): the port of the Moraiti et al. method, below Source in all four protocols (cause unknown).
