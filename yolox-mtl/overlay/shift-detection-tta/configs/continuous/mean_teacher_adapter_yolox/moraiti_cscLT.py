# Moraiti et al. MT + CL + SR (official adapter, committed Cityscapes hyper-parameters) on the Cityscapes-C
# long-term stream: (fog -> motion_blur -> snow -> brightness -> defocus_blur) x 10, no reset (50 x 500 images).
_base_ = ['../mean_teacher_adapter_yolox/yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/'
corruptions = ['fog', 'motion_blur', 'snow', 'brightness', 'defocus_blur'] * 10

model = dict(adapter=dict(plot=False, plot_augmented_imgs=False))

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
                ann_file=root + 'cityscapes/instancesonly_filtered_gtFine_val_yoloxorder.json',
                data_prefix=dict(img=root + '%s/leftImg8bit/val/' % c),
                test_mode=True,
                pipeline={{_base_.test_pipeline}},
                metainfo=dict(classes=classes)) for c in corruptions
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[500] * 50, domain_names=corruptions),
]
test_evaluator = val_evaluator
