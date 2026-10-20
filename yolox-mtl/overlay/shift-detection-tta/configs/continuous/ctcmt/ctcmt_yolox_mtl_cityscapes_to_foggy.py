"""CT-CMT — Cityscapes -> Foggy Cityscapes continual TTA.

Adapts the same source YOLOX-MTL model as ``ctcmt_yolox_mtl_cityscapes_to_acdc``
across the three fog levels (low -> medium -> high). Overlaps directly with
the Cityscapes-Fog rows of your prior single-task paper (Moraiti et al. 2026)
so per-fog-level numbers are comparable.

Foggy Cityscapes labels reuse the semantic-seg gt from the underlying
Cityscapes val set.

Data prep:
    1. Prepare clear Cityscapes (see tools/data/prepare_cityscapes.sh).
    2. Unzip leftImg8bit_trainvaltest_foggyDBF.zip so images land at
       data/cityscapes_foggy/leftImg8bit_foggyDBF/val/<city>/*_leftImg8bit_foggy_beta_{beta}.png
    3. Emit one COCO json per fog density (image `file_name` fields patched):
         python tools/data/make_foggy_cityscapes_json.py \\
             --clear-json data/cityscapes/instancesonly_filtered_gtFine_val.json \\
             --out-dir    data/cityscapes_foggy/
"""

_base_ = [
    './ctcmt_yolox_mtl_cityscapes_to_acdc.py',
]

data_root_cs = 'data/cityscapes/'
data_root_foggy = 'data/cityscapes_foggy/'
img_scale = (800, 1440)

_metainfo = dict(classes=('person', 'rider', 'car', 'train',
                          'motorcycle', 'bicycle', 'truck', 'bus'))


def _foggy_dataset(beta):
    """Foggy Cityscapes val set at fog density ``beta`` (0.005 / 0.01 / 0.02).

    Uses a per-beta json emitted by ``tools/data/make_foggy_cityscapes_json.py``
    whose image `file_name` fields already point at the foggy renderings.
    """
    return dict(
        type='mmdet.datasets.CocoDataset',
        ann_file=data_root_foggy +
                 f'instancesonly_filtered_gtFine_val_foggy_beta_{beta}.json',
        data_prefix=dict(
            img=data_root_foggy + 'leftImg8bit_foggyDBF/val/',
            seg=data_root_cs + 'gtFine/val/',
        ),
        seg_map_suffix='_gtFine_labelTrainIds.png',
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
            dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
            dict(type='mmdet.Pad', size_divisor=32,
                 pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
            dict(type='mmtrack.PackTrackInputs', pack_single_img=True),
        ],
        metainfo=_metainfo,
    )


concat_dataset = dict(
    type='mmdet.ConcatDataset',
    datasets=[
        _foggy_dataset('0.005'),   # Low fog
        _foggy_dataset('0.01'),    # Medium fog
        _foggy_dataset('0.02'),    # High fog
        # Loopback to clear Cityscapes val (matches the prior paper's protocol
        # for measuring `Forgetting` = mAP_Source - mAP_Loopback).
        dict(
            type='mmdet.datasets.CocoDataset',
            ann_file=data_root_cs + 'instancesonly_filtered_gtFine_val.json',
            data_prefix=dict(img=data_root_cs + 'leftImg8bit/val/',
                             seg=data_root_cs + 'gtFine/val/'),
            seg_map_suffix='_gtFine_labelTrainIds.png',
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='mmtrack.LoadTrackAnnotations', with_seg=True),
                dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
                dict(type='mmdet.Pad', size_divisor=32,
                     pad_val=dict(img=(114.0, 114.0, 114.0), seg=255)),
                dict(type='mmtrack.PackTrackInputs', pack_single_img=True),
            ],
            metainfo=_metainfo,
        ),
    ],
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=concat_dataset,
)
test_dataloader = val_dataloader

