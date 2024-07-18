_base_ = [
    '../_base_/models/convnext/convnext-base.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

train_cfg = dict(max_epochs=20, val_interval=10)

train_dataloader = dict(
    batch_size=2,  # 减小批量大小
    dataset=dict(
        data_root='/home/xlsk/Code/mmpretrain/configs/resnet/imagenet/',
        classes=[f'class{i}' for i in range(1, 6)]
    )
)
val_dataloader = dict(
    batch_size=4,  # 减小验证批量大小
    dataset=dict(
        data_root='/home/xlsk/Code/mmpretrain/configs/resnet/imagenet/',
        classes=[f'class{i}' for i in range(1, 6)]
    )
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        milestones=[5, 10, 15],
        gamma=0.1,
        by_epoch=True,
        begin=0,
        end=20,
    )
]

optim_wrapper = dict(
    type='AmpOptimWrapper',  # 启用混合精度训练
    optimizer=dict(type='AdamW', lr=4e-3),
    clip_grad=None,
)



auto_scale_lr = dict(base_batch_size=4096)

env_cfg = dict(
    cudnn_benchmark=True,  # 将 cudnn_benchmark 设置为 True
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0)
)

