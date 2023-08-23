_base_ = [
    'efficientnet_b0.py',
    'imagenet_bs32.py',
    'imagenet_bs256.py',
    'default_runtime.py',
]

# dataset settings
train_pipeline = [ 
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetRandomCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='EfficientNetCenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))


model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpt/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=58),
)