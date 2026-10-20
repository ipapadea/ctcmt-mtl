"""Source-only YOLOX-MTL on clean Cityscapes val (sanity check of the source checkpoint)."""
_base_ = ['../../source/yolox_mtl/yolox_x_mtl_v2_8xb4-24e_cityscapes.py']

img_scale = (800, 1440)
root = '/workspace/data/'
classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
class_names19 = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
                 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                 'truck', 'bus', 'train', 'motorcycle', 'bicycle')

val_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='SuffixSegDataset', rgb_suffix='_leftImg8bit',
        ann_file=root + 'cityscapes/instancesonly_filtered_gtFine_val.json',
        data_prefix=dict(img=root + 'cityscapes/leftImg8bit/val/',
                         seg=root + 'cityscapes/gtFine/val/'),
        seg_map_suffix='_gtFine_labelTrainIds.png', test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.Pad', size_divisor=32,
                 pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
            dict(type='PackTrackInputsWithSeg', pack_single_img=True),
        ],
        metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[500], domain_names=['cityscapes_clear_val']),
    dict(type='IoUMetricDomains', num_classes=19, ignore_index=255, iou_metrics=['mIoU'],
         class_names=class_names19, domain_sizes=[500], domain_names=['cityscapes_clear_val']),
]
test_evaluator = val_evaluator
