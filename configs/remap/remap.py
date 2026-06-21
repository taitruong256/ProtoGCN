modality = 'j'
graph = 'openpose25'
num_classes = 5
fold = 0
val_fold = (fold + 1) % 5
train_folds = [i for i in range(5) if i not in {fold, val_fold}]

work_dir = f'./work_dirs/remap/fold_{fold}'
data_root = 'data/REMAP/SitToStand'
skeleton_dir = f'{data_root}/Data/STS_2D_skeletons_coarsened'
split_dir = f'{data_root}/splits'


def _dataset_cfg(split, fold_id):
    return dict(
        type='RemapSitToStandDataset',
        ann_file=f'{split_dir}/fold_{fold_id}_{split}.csv',
        skeleton_dir=skeleton_dir,
        pipeline=train_pipeline if split == 'train' else (val_pipeline if split == 'val' else test_pipeline),
        label_file=f'{data_root}/Data/STS_human_labels/SitToStand_human_labels.xls',
    )


model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=3,
        num_person=1,
        num_prototype=300,
        use_view=False,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02),
    ),
    cls_head=dict(
        type='SimpleHead',
        joint_cfg=graph,
        num_classes=num_classes,
        in_channels=384,
        weight=0.2,
    ),
    view_loss_weight=0.0,
    test_cfg=dict(feat_ext=False),
)

train_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=10),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='ConcatDataset',
        datasets=[_dataset_cfg('train', i) for i in train_folds],
    ),
    val=_dataset_cfg('val', val_fold),
    test=_dataset_cfg('test', fold),
)

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 200
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=1,
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
    save_best='f1_score',
    rule='greater',
)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
