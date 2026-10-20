# Moraiti et al. MT + CL + SR (official adapter, committed Cityscapes hyper-parameters) on the ACDC long-term
# stream: the ACDC4 cycle (fog -> night -> rain -> snow) repeated 10 times, no reset (40 x 400 images).
_base_ = ['../mean_teacher_adapter_yolox/yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/ACDC/'
weathers = ['fog', 'night', 'rain', 'snow'] * 10

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
                ann_file=root + 'gt_detection/%s/instancesonly_%s_train_gt_detection_yoloxorder.json' % (w, w),
                data_prefix=dict(img=root + 'rgb_anon/'),
                test_mode=True,
                pipeline={{_base_.test_pipeline}},
                metainfo=dict(classes=classes)) for w in weathers
        ]))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=False,
         domain_sizes=[400] * 40, domain_names=weathers),
]
test_evaluator = val_evaluator
