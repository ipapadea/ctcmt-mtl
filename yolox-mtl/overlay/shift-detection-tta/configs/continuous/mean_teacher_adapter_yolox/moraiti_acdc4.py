# Mean Teacher + object-level contrastive learning + stochastic restoration (Moraiti et al., EJAI 2025),
# official adapter code (MeanTeacherYOLOXAdapterContrastive) with the hyper-parameters of the committed
# Cityscapes config, on the ACDC4 stream (fog -> night -> rain -> snow, train split, 400 images each).
#
# Same data / class-order handling as source_only_acdc4.py. Continual, batch size 1, no reset (the images
# have no frame_id, so `episodic` never fires).  optim_steps is 1 as in the committed Cityscapes config; the
# paper text says 5 (use --cfg-options model.adapter.optim_steps=5).
_base_ = ['../mean_teacher_adapter_yolox/yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/ACDC/'
weathers = ['fog', 'night', 'rain', 'snow']

# the adapter would write loss plots to results/plots (the repo is mounted read-only)
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
         domain_sizes=[400] * 4, domain_names=weathers),
]
test_evaluator = val_evaluator
