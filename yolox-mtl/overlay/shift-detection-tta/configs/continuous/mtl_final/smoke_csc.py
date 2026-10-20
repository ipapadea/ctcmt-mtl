"""Smoke test: first 100 frames of CSC12 (defocus_blur), adaptation on, per-domain metrics."""
_base_ = ['./source_only_csc12.py']

val_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dict(
            type='SuffixSegDataset', rgb_suffix='_leftImg8bit', indices=100,
            ann_file='/workspace/data/cityscapes/instancesonly_filtered_gtFine_val.json',
            data_prefix=dict(img='/workspace/data/defocus_blur/leftImg8bit/val/',
                             seg='/workspace/data/cityscapes/gtFine/val/'),
            seg_map_suffix='_gtFine_labelTrainIds.png', test_mode=True,
            pipeline={{_base_.eval_pipeline}}, metainfo=dict(classes={{_base_.classes}}))]))
test_dataloader = val_dataloader
val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False, domain_sizes=[100],
         domain_names=['defocus100']),
    dict(type='IoUMetricDomains', num_classes=19, ignore_index=255, iou_metrics=['mIoU'],
         class_names={{_base_.class_names19}}, domain_sizes=[100], domain_names=['defocus100']),
]
test_evaluator = val_evaluator
