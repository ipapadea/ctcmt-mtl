# Source-only YOLOX-X (no adaptation) on the Cityscapes-C 12-corruption stream (CSC12),
# 500 images per corruption, in the order used by the AMROD / CTCMT experiments.
# Corruption images live in <corruption>/leftImg8bit/val/<city>/..., the detection json is
# the clean Cityscapes val one (corrupted images keep the original file names).
_base_ = ['./yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/'
corruptions = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost',
               'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']

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
