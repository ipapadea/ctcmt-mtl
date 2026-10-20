"""CT-CMT-MTL full XVA — Cityscapes -> ACDC continual TTA.

Extends ctcmt_yolox_mtl_cityscapes_to_acdc.py with all four XVA components
enabled (METHOD.md §3-§5.5):
  - CT-CL  on student features         (§2, always on)
  - CT-CR  dense bbox→seg-class CE     (§4, always on, ct_cr_weight=0.05)
  - STR    backbone/neck restore η=0.1 (§5, backbone_rst_factor=0.1 default)
  - CTPV   cross-task pseudo-label veto (§5.5, enabled here)
  - Score-EM gate                      (§6, score_em_init=0.5 default)

Run with:
    python tools/test.py \\
        configs/continuous/ctcmt/ctcmt_yolox_mtl_cityscapes_to_acdc_full_xva.py \\
        --checkpoint /data/ilias/yolox-mtl_weights/yolox_x_cs_mtl_v2_epoch24.pth \\
        --work-dir work_dirs/ctcmt_mtl_full_xva_acdc_seed0 \\
        --cfg-options randomness.seed=0
"""

_base_ = ['./ctcmt_yolox_mtl_cityscapes_to_acdc.py']

model = dict(
    adapter=dict(
        # CTPV: reject det pseudo-boxes not corroborated by seg head (§5.5).
        # det_to_seg_cls maps ACDC det index -> Cityscapes seg trainId.
        # ACDC det order: person=0, rider=1, car=2, train=3, motorcycle=4,
        #                 bicycle=5, truck=6, bus=7
        ctpv_enabled=True,
        ctpv_thresh=0.3,
        det_to_seg_cls=(11, 12, 13, 16, 17, 18, 14, 15),
        # STR shared-trunk factor η (already default=0.1, explicit for clarity)
        backbone_rst_factor=0.1,
        # Score-EM gate (already defaults, explicit for paper traceability)
        score_em_init=0.5,
        score_gamma=0.7,
        score_thresh=1.4,
    )
)
