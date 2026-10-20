"""MTL v2: YOLOX-X + semantic segmentation head on Cityscapes.

Key fixes over v1:
  - NO Mosaic/MixUp augmentation.  MixUp alpha-blends pixel values from two
    images while keeping only one image's dense seg labels, creating systematic
    wrong supervision.  The seg-only baseline deliberately avoids these
    augmentations for the same reason (see its comment).
  - Class-weighted CE using Cityscapes median-frequency balancing weights so
    rare classes (train, motorcycle, rider, traffic_light) receive proportionally
    stronger gradient signal instead of being drowned out by road/building.
  - Logits upsampled to input resolution for the CE loss — better boundary
    supervision than downsampling the GT mask to stride-8.
  - Larger seg head: hidden_channels 128 → 256, num_convs_per_level 2 → 3,
    fusion 'add' → 'cat' for richer multi-scale representation.
  - Seg loss_weight 1.0 → 2.0 so the seg gradient has comparable influence on
    the shared backbone as the combined detection losses.

Detection augmentation: RandomFlip + PhotoMetricDistortion + BatchSyncRandomResize.
This is the same set used by the seg-only baseline and is fully seg-compatible.
"""

_base_ = [
    '../../_base_/default_runtime.py',
]

dataset_type = 'CityscapesSegDataset'
data_root = 'data/cityscapes/'
attributes = dict(weather_coarse='all', timeofday_coarse='all')

img_scale = (800, 1440)
batch_size = 2
num_seg_classes = 19

# Cityscapes 19-class median-frequency balancing weights.
# freq[c] = fraction of valid pixels belonging to class c (Cityscapes train).
# weight[c] = median(freq) / freq[c], clipped to avoid extreme boosts.
# Source: ENet (Paszke et al. 2016); widely used in Cityscapes seg literature.
_cityscapes_class_weights = [
    0.0228,  # road
    0.1399,  # sidewalk
    0.0368,  # building
    1.2460,  # wall
    1.0000,  # fence  (median class)
    0.6270,  # pole
    2.9410,  # traffic_light
    1.4330,  # traffic_sign
    0.0559,  # vegetation
    0.5630,  # terrain
    0.1508,  # sky
    0.7170,  # person
    4.3130,  # rider
    0.1145,  # car
    2.0170,  # truck
    1.5660,  # bus
    7.2500,  # train
    5.7600,  # motorcycle
    1.2350,  # bicycle
]

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
            hidden_channels=256,
            num_convs_per_level=3,
            fusion='cat',
            ignore_index=255,
            loss_weight=2.0,
            dropout_p=0.1,
            class_weights=_cityscapes_class_weights),
        seg_ignore_index=255),
    adapter=None)

# No Mosaic/MixUp: these augmentations alpha-blend pixel values between two
# images while keeping only one image's dense seg labels, creating incorrect
# supervision for the segmentation head.  Use the same seg-safe pipeline as
# the seg-only baseline.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PhotoMetricDistortion'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackTrackInputsWithSeg', pack_single_img=True),
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
    type=dataset_type,
    ann_file=data_root + 'instancesonly_filtered_gtFine_train.json',
    data_prefix=dict(
        img=data_root + 'leftImg8bit/train/',
        seg=data_root + 'gtFine/train/',
    ),
    seg_map_suffix='_gtFine_labelTrainIds.png',
    metainfo=dict(classes=('person', 'rider', 'car', 'train',
                           'motorcycle', 'bicycle', 'truck', 'bus')),
    pipeline=train_pipeline,
    filter_cfg=dict(
        attributes=attributes,
        filter_empty_gt=False,
        min_size=32
    ))
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
    dict(type='IoUMetric', num_classes=19, ignore_index=255,
         iou_metrics=['mIoU'],
         class_names=('road', 'sidewalk', 'building', 'wall', 'fence',
                       'pole', 'traffic_light', 'traffic_sign', 'vegetation',
                       'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                       'bus', 'train', 'motorcycle', 'bicycle')),
]
test_evaluator = val_evaluator
