"""Smoke test: first 80 fog frames of ACDC (adaptation on, per-domain metrics)."""
_base_ = ['./source_only_acdc4.py']

val_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dict(
            type='SuffixSegDataset', rgb_suffix='_rgb_anon', indices=80,
            ann_file='/workspace/data/ACDC/gt_detection/fog/instancesonly_fog_train_gt_detection.json',
            data_prefix=dict(img='/workspace/data/ACDC/rgb_anon/', seg='/workspace/data/ACDC/gt/'),
            seg_map_suffix='_gt_labelTrainIds.png', test_mode=True,
            pipeline={{_base_.eval_pipeline}}, metainfo=dict(classes={{_base_.classes}}))]))
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False, domain_sizes=[80],
         domain_names=['fog80']),
    dict(type='IoUMetricDomains', num_classes=19, ignore_index=255, iou_metrics=['mIoU'],
         class_names={{_base_.class_names19}}, domain_sizes=[80], domain_names=['fog80']),
]
test_evaluator = val_evaluator
