# Source-only YOLOX-X (no adaptation) on the ACDC4 stream of the benchmark
# (fog -> night -> rain -> snow, train split, 400 images per condition).
# Detection jsons are the same ones used by the AMROD / CTCMT experiments.
_base_ = ['./yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/ACDC/'
weathers = ['fog', 'night', 'rain', 'snow']

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='mmdet.datasets.CocoDataset',
                ann_file=root + 'gt_detection/%s/instancesonly_%s_train_gt_detection_yoloxorder.json' % (w, w),
                data_prefix=dict(img=root + 'rgb_anon/'),
                test_mode=True,
                pipeline={{_base_.test_pipeline}},
                metainfo=dict(classes=classes)) for w in weathers
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[400, 400, 400, 400], domain_names=weathers),
]
test_evaluator = val_evaluator
