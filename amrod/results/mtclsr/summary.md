# MTCLSR: same-source port of Moraiti et al. (MT + CL + SR) to Panoptic-FPN

Detection branch only. Paper hyper-parameters, untuned. Seed 0, one run per protocol. AP50 = mean over the
domains of the stream. Config: `detectron2/configs/Cityscapes/mtclsr_pfn_R_50_{ACDC,CS_C}.yaml`,
runner: `scripts/paper26_batch_mtclsr.sh`, meta-architecture: `detectron2/detectron2/modeling/meta_arch/mtclsr.py`.

| Protocol | Source AP50 | MTCLSR AP50 |
|---|---|---|
| ACDC4 | 34.08 | 33.82 |
| CSC12 | 13.46 | 13.28 |
| ACDC-LT | 34.08 (repeated) | 27.16 |
| CSC-LT | not tabulated (Source is not adapted, so it equals its CSC12 per-domain scores for the 5 domains of the stream, repeated) | 9.90 |

Mean AP50 per round of the long-term runs (it falls monotonically):

- ACDC-LT (4 domains per round): 33.9, 31.6, 28.9, 27.3, 26.3, 25.7, 25.0, 24.7, 24.1, 24.1
- CSC-LT (5 domains per round): 15.2, 13.8, 12.9, 12.0, 10.9, 9.7, 8.0, 6.4, 5.4, 4.7

The port is below Source in all four protocols. It is **not known** whether this is a bug of the port, an effect of the
architecture (the objectness / RPN / ROI heads of Faster R-CNN instead of YOLOX) or a hyper-parameter mismatch; the checks
"lr = 0 reproduces Source" and "optim_steps = 1" were not run. Do not read this row as a result about the original method:
the original method on its own YOLOX-X checkpoint is in `yolox-mtl/results/yolox_moraiti/`.

`logs/` contains the run logs (gzip) and the run metadata (`*.txt`). The run metadata records `a58ba2f`,
the parent of the commit that adds `mtclsr.py` (the file was not committed yet when the runs started).
