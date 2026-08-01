import os


graph = 'smpl_24'
num_classes = 4
fold_id = int(os.environ.get('CARE_PD_FOLD', 1))
# Keep this separate from checkpoints created by the former learnable-fusion
# architecture; tools/train.py auto-resumes from work_dir/latest.pth.
work_dir = f'./work_dirs/care_pd/pd_gam_4branch_ensemble/fold_{fold_id}'


def make_backbone(in_channels):
    return dict(
        type='ProtoGCN',
        in_channels=in_channels,
        num_prototype=300,
        gcn_use_view_graph=False,
        base_channels=64,
        num_stages=4,
        inflate_stages=[3, 4],
        down_stages=[3, 4],
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(
            layout=graph, mode='random', num_filter=8,
            init_off=.04, init_std=.02))


model = dict(
    type='MultiBranchRecognizerGCN',
    branches=[
        dict(name='joint', in_channels=3, backbone=make_backbone(3)),
        dict(name='joint_motion', in_channels=3, backbone=make_backbone(3)),
        dict(name='angle', in_channels=1, backbone=make_backbone(1)),
        dict(name='bone', in_channels=3, backbone=make_backbone(3)),
    ],
    cls_head=dict(
        type='MultiBranchHead',
        joint_cfg=graph,
        num_classes=num_classes,
        branch_channels=[256, 256, 256, 256],
        branch_names=['joint', 'joint_motion', 'angle', 'bone'],
        ensemble_weights=[1, 1, 1, 1],
        weight=0.2),
    test_cfg=dict(average_clips='prob'))


dataset_type = 'CarePDDataset'
ann_file = 'data/CARE-PD/folds/UPDRS_Datasets/PD-GaM_6fold_participants_skeleton.pkl'

common_pipeline = [
    dict(type='PreNormalize3D'),
    # Merge order determines the branch slices:
    # joint(3), joint_motion(3), angle(1), bone(3).
    dict(type='GenSkeFeat', dataset=graph, feats=['j', 'jm', 'a', 'b']),
]

train_pipeline = common_pipeline + [
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'],
         meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = common_pipeline + [
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'],
         meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = common_pipeline + [
    dict(type='UniformSampleDecode', clip_len=100, num_clips=10),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'],
         meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    # Four full ProtoGCN branches require more memory than a single stream.
    videos_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type, ann_file=ann_file, fold=fold_id,
        split='train', pipeline=train_pipeline),
    val=dict(
        type=dataset_type, ann_file=ann_file, fold=fold_id,
        split='eval', pipeline=val_pipeline, test_mode=True),
    test=dict(
        type=dataset_type, ann_file=ann_file, fold=fold_id,
        split='eval', pipeline=test_pipeline, test_mode=True))

optimizer = dict(type='SGD', lr=0.025, momentum=0.9,
                 weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=10, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=10,
    metrics=['accuracy', 'f1_score', 'precision', 'recall'],
    save_best='f1_score', rule='greater')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
