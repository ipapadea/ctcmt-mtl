"""Source-only (Base) YOLOX-MTL on CSC12: 12 Cityscapes-C corruptions x 500 images (val).

Corrupted images live in ``<corruption>/leftImg8bit/val/<city>/`` and keep the clean file names, so the
clean Cityscapes val detection json and ``gtFine`` trainId masks are reused. Per-domain metrics averaged
over domains (AMROD protocol).
"""
_base_ = ['../../source/yolox_mtl/yolox_x_mtl_v2_8xb4-24e_cityscapes.py']

img_scale = (800, 1440)
root = '/workspace/data/'
corruptions = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
               'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']
classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
class_names19 = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='mmdet.Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
    dict(type='PackTrackInputsWithSeg', pack_single_img=True),
]

val_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='SuffixSegDataset', rgb_suffix='_leftImg8bit',
                ann_file=root + 'cityscapes/instancesonly_filtered_gtFine_val.json',
                data_prefix=dict(img=root + '%s/leftImg8bit/val/' % c,
                                 seg=root + 'cityscapes/gtFine/val/'),
                seg_map_suffix='_gtFine_labelTrainIds.png', test_mode=True,
                pipeline=eval_pipeline, metainfo=dict(classes=classes)) for c in corruptions
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[500] * 12, domain_names=corruptions),
    dict(type='IoUMetricDomains', num_classes=19, ignore_index=255, iou_metrics=['mIoU'],
         class_names=class_names19, domain_sizes=[500] * 12, domain_names=corruptions),
]
test_evaluator = val_evaluator
