# Source-only YOLOX-X (no adaptation) on clean Cityscapes val: sanity check of a source
# checkpoint against the "No adaptation / Clear" mAP of the paper (Table 5).
# Uses the Cityscapes val detection json of the AMROD benchmark (8 classes) with its categories
# re-ordered to the class order of the checkpoint (tools/data/reorder_coco_categories.py): mmdet
# derives the labels from the json's category order, not from metainfo.classes.
#
#   python tools/test.py configs/continuous/no_adap_yolox/source_only_cityscapes_val.py \
#       --checkpoint <yolox_x_8xb4-24e_cityscapes.pth>
_base_ = ['./yolox_x_8xb4-24e_cityscapes.py']

classes = ('person', 'rider', 'car', 'train', 'motorcycle', 'bicycle', 'truck', 'bus')
root = '/workspace/data/'

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmdet.datasets.CocoDataset',
        ann_file=root + 'cityscapes/instancesonly_filtered_gtFine_val_yoloxorder.json',
        data_prefix=dict(img=root + 'cityscapes/leftImg8bit/val/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}},
        metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='CocoMetricDomains', metric=['bbox'], classwise=True,
         domain_sizes=[500], domain_names=['cityscapes_clear_val']),
]
test_evaluator = val_evaluator
