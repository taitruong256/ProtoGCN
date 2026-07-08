modality = 'j'
graph = 'coco'
work_dir = f'./work_dirs/casia_b/j_triplet'
auto_resume = False
log_model_complexity = False

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=3,
        num_prototype=300,
        base_channels=64,
        gcn_use_view=False,
        gcn_use_euclid=False,
        use_bottleneck=False,
        block_variant='split_residual_v1',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=None,
    train_cfg=dict(
        triplet_loss=dict(type='TripletLoss', margin=0.3, is_hard_loss=True)),
    test_cfg=dict(feat_ext=True, pool_opt='nmtv'))

dataset_type = 'CasiaBGaitDataset'
train_ann_file = 'data/casia-b/casia-b_pose_train_valid.csv'
val_ann_file = 'data/casia-b/casia-b_pose_valid.csv'
test_ann_file = 'data/casia-b/casia-b_pose_test.csv'
train_pipeline = [
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=60),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=60, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=60, num_clips=10),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=0,
    train_dataloader=dict(
        triplet_sampler=dict(
            batch_size=[4, 32],
            batch_shuffle=False)),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=train_ann_file, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=val_ann_file, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=test_ann_file, pipeline=test_pipeline))

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
runner_type = 'IterBasedRunner'
total_iters = 40000
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=1000,
    by_epoch=False,
    metrics=['gait_rank1', 'gait_contrastive_loss'],
    save_best='gait_contrastive_loss',
    rule='less')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
