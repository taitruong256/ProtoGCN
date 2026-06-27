modality = 'j'
graph = 'coco'
num_classes = 74 
work_dir = f'./work_dirs/casia_b/j_mini'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=3,
        num_prototype=300,
        view_num=11,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='SimpleHead', joint_cfg=graph, num_classes=num_classes, in_channels=384, weight=0.2),
    view_loss_weight=1.0,
    test_cfg=dict(feat_ext=True, pool_opt='nmtv'))

dataset_type = 'CasiaBGaitDataset'
train_ann_file = 'data/casia-b/casia-b_pose_train.csv'
val_ann_file = 'data/casia-b/casia-b_pose_valid.csv'
test_ann_file = 'data/casia-b/casia-b_pose_test.csv'
mini_views = (0, 36, 72, 108, 144, 180)
mini_subject_ratio = 0.5
train_pipeline = [
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'condition', 'view', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        mini_views=mini_views,
        mini_subject_ratio=mini_subject_ratio),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        pipeline=val_pipeline,
        mini_views=mini_views,
        mini_subject_ratio=mini_subject_ratio),
    test=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        pipeline=test_pipeline,
        mini_views=mini_views,
        mini_subject_ratio=mini_subject_ratio))

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 50
checkpoint_config = dict(interval=5, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=5,
    metrics=['gait_rank1', 'gait_contrastive_loss'],
    save_best='gait_contrastive_loss',
    rule='less')
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
