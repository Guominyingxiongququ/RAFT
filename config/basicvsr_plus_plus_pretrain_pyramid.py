exp_name = 'basicvsr_plusplus_pretrain_pyramid'

# model settings
model = dict(
    type='BasicVSRDN',
    generator=dict(
        type='BasicVSRPlusPlusDN',
        mid_channels=64,
        num_blocks=7,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_dataset_type = 'DNTestDataset'
val_dataset_type = 'DNTestDataset'

train_pipeline = [
    dict(type='GenerateDNFrameIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateDNTestFrameIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train= dict(type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_noisy',
            gt_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_gt',
            ann_file='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_gt/'
            'meta_info_CRVD_GT.txt',
            num_input_frames=5,
            pipeline=train_pipeline,
            scale=1,
            val_partition='CRVD',
            test_mode=True)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_noisy',
        gt_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_gt',
        ann_file='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_gt/'
        'meta_info_CRVD_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=1,
        val_partition='CRVD',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_noisy',
        gt_folder='/home/xinyuanyu/data/CRVD_dataset/indoor_rgb_gt',
        ann_file='/home/xinyuanyu/data/CRVD_dataset'
        '/indoor_rgb_gt/meta_info_CRVD_GT.txt',
        num_input_frames=5,
        pipeline=test_pipeline,
        scale=1,
        val_partition='CRVD',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 200000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[200000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/home/xinyuanyu/new_result/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
