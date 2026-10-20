# Moraiti et al. MT + CL + SR (official adapter, committed Cityscapes hyper-parameters) on the Cityscapes-C
# 12-corruption stream (CSC12, 500 images each, order of the AMROD / CTCMT experiments).
# Same data / class-order handling as source_only_csc12.py.
_base_ = ['../mean_teacher_adapter_yolox/yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/'
corruptions = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
               'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']

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
         domain_sizes=[500] * 12, domain_names=corruptions),
]
test_evaluator = val_evaluator
