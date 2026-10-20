"""CT-CMT (Cross-Task Contrastive Mean Teacher) — Cityscapes -> ACDC continual TTA.

Streams the ACDC val set through the CTTA weather cycle
(fog -> night -> rain -> snow) and adapts the source YOLOX-MTL model with our
cross-task contrastive mean-teacher method.

Points to the SOURCE checkpoint produced by
``configs/source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py`` (override
``load_from`` on the CLI).
"""

_base_ = [
    '../../source/yolox_mtl/yolox_x_mtl_8xb4-24e_cityscapes.py',
]

data_root_acdc = 'data/acdc/'
img_scale = (800, 1440)

# The source config's evaluator is inherited; we swap datasets to ACDC.

# ---- Adapter ----------------------------------------------------------
model = dict(
    adapter=dict(
        _delete_=True,   # source config has `adapter=None`; replace, don't merge.
        type='MeanTeacherYOLOXMTLAdapterCTCMT',
        teacher=dict(
            type='mmdet.ExpMomentumEMA',
            momentum=1 - 0.99965,   # matches your paper (alpha=0.99965)
            update_buffers=True,
        ),
        fixed_source_model=True,
        stochastic_restoration=True,
        rst_prob=0.01,
        task_weighted_restore=False,   # ablation toggle
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(type='SGD', lr=0.00025, momentum=0.9,
                           weight_decay=0.0005, nesterov=True),
            paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)),
        optim_steps=5,
        views=2,
        # `pipeline` runs once per sample and only loads the raw image; teacher
        # and student pipelines then Resize/Pad/Pack it into a TrackDataSample.
        pipeline=[dict(type='LoadImageFromFile')],
        teacher_pipeline=[
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.Pad', size_divisor=32,
                 pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='mmtrack.PackTrackInputs', pack_single_img=True),
        ],
        student_pipeline=[
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(type='mmdet.Pad', size_divisor=32,
                 pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='mmtrack.PackTrackInputs', pack_single_img=True),
        ],
        # Loss configs -----------------------------------------------------
        mtl_consistency_loss=dict(
            type='MTLConsistencyLoss',
            weight_det=0.005,
            weight_seg=1.0,
            obj_weight=1.0, reg_weight=1.0, cls_weight=1.0,
            seg_temperature=1.0,
        ),
        ct_cl_loss=dict(
            type='CrossTaskContrastiveLoss',
            weight=0.003,
            temperature=0.07,
            roi_output_size=(7, 7),
            num_scales=3,
            include_seg_view=True,
        ),
        ct_cr_weight=0.05,
        spatial_scales=(1 / 8, 1 / 16, 1 / 32),
        seg_num_classes=19,
        det_conf_threshold=0.7,
        filter_pseudo_labels=0.7,
    )
)

# ---- ACDC target datasets (weather-ordered) ---------------------------
_acdc_metainfo = dict(classes=('person', 'rider', 'car', 'train',
                               'motorcycle', 'bicycle', 'truck', 'bus'))

def _acdc_dataset(weather, split='val'):
    return dict(
        type='mmdet.datasets.CocoDataset',
        ann_file=data_root_acdc + f'gt_detection/{weather}/'
                 f'instancesonly_{weather}_{split}_gt_detection.json',
        data_prefix=dict(img=data_root_acdc + 'rgb_anon/',
                         seg=data_root_acdc + 'gt/'),
        seg_map_suffix='_gt_labelTrainIds.png',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.Pad', size_divisor=32,
                 pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
            dict(type='mmtrack.PackTrackInputs', pack_single_img=True),
        ],
        metainfo=_acdc_metainfo,
    )


# Concatenate the four weathers in CoTTA/ACDC canonical order.
concat_dataset = dict(
    type='mmdet.ConcatDataset',
    datasets=[
        _acdc_dataset('fog'),
        _acdc_dataset('night'),
        _acdc_dataset('rain'),
        _acdc_dataset('snow'),
    ],
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=concat_dataset,
)
test_dataloader = val_dataloader

# Detection + segmentation eval:
val_evaluator = [
    dict(type='mmdet.CocoMetric', metric=['bbox'], classwise=True),
    dict(type='IoUMetric', num_classes=19, ignore_index=255, iou_metrics=['mIoU'], class_names=('road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')),
]
test_evaluator = val_evaluator

# Number of adaptation cycles (CoTTA convention = 10; use 1 for a first pass).
num_repeats = 1

# Reset teacher to source at every new weather block? (CoTTA reports "no",
# your prior paper's SHIFT results tested both.) Off by default.
reset_between_weathers = False
