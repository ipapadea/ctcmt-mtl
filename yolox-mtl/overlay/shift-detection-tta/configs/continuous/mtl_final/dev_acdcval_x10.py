"""Development stream x10 (4060 frames, stability check): ACDC *val* split (fog 100, night 106, rain 100, snow 100), disjoint from the ACDC4
benchmark stream (train split). Used only to choose optimiser hyper-parameters of the YOLOX adapter.

Detection and segmentation are scored per domain (CocoMetricDomains / IoUMetricDomains) and averaged
over domains, as in the AMROD tables. Paths are those seen inside docker (``/workspace/data``).
Class order of the detection json (person, rider, car, truck, bus, train, motorcycle, bicycle)
is the one the source model was trained with.
"""
_base_ = ['../../source/yolox_mtl/yolox_x_mtl_v2_8xb4-24e_cityscapes.py']

img_scale = (800, 1440)
root = '/workspace/data/'
weathers = ['fog', 'night', 'rain', 'snow'] * 10
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
                type='SuffixSegDataset', rgb_suffix='_rgb_anon',
                ann_file=root + 'ACDC/gt_detection/%s/instancesonly_%s_val_gt_detection.json' % (w, w),
                data_prefix=dict(img=root + 'ACDC/rgb_anon/', seg=root + 'ACDC/gt/'),
                seg_map_suffix='_gt_labelTrainIds.png', test_mode=True,
                pipeline=eval_pipeline, metainfo=dict(classes=classes)) for w in weathers
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[100, 106, 100, 100] * 10, domain_names=weathers),
    dict(type='IoUMetricDomains', num_classes=19, ignore_index=255, iou_metrics=['mIoU'],
         class_names=class_names19, domain_sizes=[100, 106, 100, 100] * 10, domain_names=weathers),
]
test_evaluator = val_evaluator

model = dict(adapter=dict(
    _delete_=True, type='MeanTeacherYOLOXMTLAdapterCTCMTFinal', weight_ctcr=0.3, threshold_max=0.9))
