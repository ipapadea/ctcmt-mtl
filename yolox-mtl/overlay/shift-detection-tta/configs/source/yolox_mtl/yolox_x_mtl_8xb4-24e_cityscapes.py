"""Source training config: YOLOX-X + semantic segmentation head, jointly on
Cityscapes detection + semantic seg.

Extends ``configs/source/yolox/yolox_x_8xb4-24e_cityscapes.py`` by:
  - Swapping the detector into :class:`YOLOXMTLDetector` (YOLOX + seg head).
  - Loading Cityscapes semantic segmentation gt alongside detection gt.
  - Adding ``mmdet.LoadAnnotations`` with ``with_seg=True`` in both pipelines.

Uses the same img_scale (800, 1440), batch size, LR, and schedule as the prior
single-task config so numbers are directly comparable.
"""

_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/default_runtime.py',
]

dataset_type = 'CityscapesSegDataset'
data_root = 'data/cityscapes/'
attributes = dict(weather_coarse='all', timeofday_coarse='all')

img_scale = (800, 1440)
batch_size = 2
num_seg_classes = 19

model = dict(
    type='AdaptiveDetector',
    data_preprocessor=dict(
        type='mmtrack.TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        _delete_=True,   # base sets _scope_='mmdet'; we need shift_tta scope.
        type='YOLOXMTLDetector',
        yolox=dict(
            _scope_='mmdet',
            type='YOLOX',
            backbone=dict(
                type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
            neck=dict(
                type='YOLOXPAFPN',
                in_channels=[320, 640, 1280],
                out_channels=320,
                num_csp_blocks=4),
            bbox_head=dict(
                type='YOLOXHead',
                num_classes=8,
                in_channels=320,
                feat_channels=320),
            train_cfg=dict(
                assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
            test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
            init_cfg=dict(
                type='Pretrained',
                checkpoint=  # noqa
                'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
            )),
        seg_head=dict(
            type='YOLOXSemSegHead',
            in_channels=(320, 320, 320),
            num_classes=num_seg_classes,
            hidden_channels=128,
            num_convs_per_level=2,
            fusion='add',
            ignore_index=255,
            loss_weight=1.0,
            dropout_p=0.1),
        seg_ignore_index=255),
    adapter=None)

train_pipeline = [
    dict(
        type='mmdet.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='mmdet.MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackTrackInputsWithSeg', pack_single_img=True)
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    dict(type='PackTrackInputsWithSeg', pack_single_img=True)
]

train_dataset = dict(
    type='mmdet.MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'instancesonly_filtered_gtFine_train.json',
        data_prefix=dict(
            img=data_root + 'leftImg8bit/train/',
            seg=data_root + 'gtFine/train/',
        ),
        seg_map_suffix='_gtFine_labelTrainIds.png',
        metainfo=dict(classes=('person', 'rider', 'car', 'train',
                               'motorcycle', 'bicycle', 'truck', 'bus')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
        ],
        filter_cfg=dict(
            attributes=attributes,
            filter_empty_gt=False,
            min_size=32
        )),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataset = dict(
    type=dataset_type,
    ann_file=data_root + 'instancesonly_filtered_gtFine_val.json',
    data_prefix=dict(
        img=data_root + 'leftImg8bit/val/',
        seg=data_root + 'gtFine/val/',
    ),
    seg_map_suffix='_gtFine_labelTrainIds.png',
    test_mode=True,
    filter_cfg=dict(attributes=attributes),
    pipeline=test_pipeline,
    metainfo=dict(classes=('person', 'rider', 'car', 'train',
                           'motorcycle', 'bicycle', 'truck', 'bus')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler'),
    dataset=val_dataset)
test_dataloader = val_dataloader

lr = 0.0005 / 8 * batch_size
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

total_epochs = 24
num_last_epochs = 2
resume_from = None
interval = 2

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='mmdet.CosineAnnealingLR', eta_min=lr * 0.05, begin=1,
         T_max=total_epochs - num_last_epochs,
         end=total_epochs - num_last_epochs, by_epoch=True,
         convert_to_iter_based=True),
    dict(type='mmdet.ConstantLR', by_epoch=True, factor=1,
         begin=total_epochs - num_last_epochs, end=total_epochs),
]

custom_hooks = [
    dict(type='YOLOXMTLModeSwitchHook',
         num_last_epochs=num_last_epochs, priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(type='mmdet.EMAHook',
         ema_type='mmdet.ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         priority=49),
]
default_hooks = dict(checkpoint=dict(interval=1))

val_evaluator = [
    dict(type='mmdet.CocoMetric', metric=['bbox'], classwise=True),
    dict(type='IoUMetric', num_classes=19, ignore_index=255, iou_metrics=['mIoU'], class_names=('road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')),
]
test_evaluator = val_evaluator
