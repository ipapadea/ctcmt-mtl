# Pre-update vs post-update teacher prediction

CTCMT and Base, 3 seeds (0, 42, 123). Mean ± sample std over seeds of the per-run domain mean.
post = adapt-then-predict (reported runs); pre = predict-then-adapt (`SOLVER.CTCMT_PREDICT_BEFORE_UPDATE=True`).

| Protocol | Method | Post AP50 | Pre AP50 | Δ AP50 (pre−post) | Post mIoU | Pre mIoU | Δ mIoU (pre−post) |
|---|---|---|---|---|---|---|---|
| ACDC4 | CTCMT | 38.007 ± 0.254 | 37.968 ± 0.123 | -0.039 | 37.800 ± 0.064 | 37.828 ± 0.045 | +0.028 |
| ACDC4 | Base | 36.839 ± 0.072 | 36.960 ± 0.127 | +0.121 | 36.864 ± 0.172 | 36.893 ± 0.161 | +0.029 |
| CSC12 | CTCMT | 17.722 ± 0.264 | 17.853 ± 0.326 | +0.131 | 30.073 ± 0.189 | 30.171 ± 0.235 | +0.098 |
| CSC12 | Base | 16.957 ± 0.291 | 16.911 ± 0.157 | -0.046 | 28.948 ± 0.665 | 29.073 ± 0.449 | +0.125 |

## Per-seed Δ (pre − post)

| Protocol | Method | Δ AP50 per seed | Δ mIoU per seed |
|---|---|---|---|
| ACDC4 | CTCMT | -0.19, -0.08, +0.16 | +0.09, +0.05, -0.05 |
| ACDC4 | Base | -0.07, +0.32, +0.11 | +0.02, +0.01, +0.05 |
| CSC12 | CTCMT | -0.03, -0.12, +0.54 | -0.12, +0.10, +0.31 |
| CSC12 | Base | +0.25, -0.20, -0.19 | +0.22, +0.29, -0.13 |

## CTCMT − Base

| Protocol | Prediction | Δ AP50 | Δ mIoU |
|---|---|---|---|
| ACDC4 | post | +1.168 | +0.936 |
| ACDC4 | pre | +1.008 | +0.935 |
| CSC12 | post | +0.765 | +1.124 |
| CSC12 | pre | +0.942 | +1.098 |

## Provenance

- Pre-update runs: `scripts/paper26_batch_preupdate.sh`; run metadata copied to `meta/`.
- The flag was added to `ctcmt_mtl.py` / `defaults.py` and was uncommitted when these runs were executed
  (`git_dirty` in the meta files); record the commit that contains it before citing the numbers.
- Short-term protocols only (ACDC4, CSC12); no long-term pre-update runs.
