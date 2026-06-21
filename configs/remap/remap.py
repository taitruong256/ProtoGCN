modality = 'j'
graph = 'openpose25'
num_classes = 4
fold = 0
exp_version = 'ver0'
use_class_weight = True
use_class_sampling = True
class_weight_power = 1.0
class_sampling_power = 0.5

work_dir = f'./work_dirs/remap/{exp_version}/fold_{fold}'
data_root = 'data/REMAP/SitToStand'
skeleton_dir = f'{data_root}/Data/STS_2D_skeletons_coarsened'
split_dir = f'{data_root}/splits'


def _dataset_cfg(split, fold_id, pipeline=None):
    return dict(
        type='RemapSitToStandDataset',
        ann_file=f'{split_dir}/fold_{fold_id}_{split}.csv',
        skeleton_dir=skeleton_dir,
        pipeline=pipeline or (train_pipeline if split == 'train' else (val_pipeline if split == 'val' else test_pipeline)),
        label_file=f'{data_root}/Data/STS_human_labels/SitToStand_human_labels.xls',
        num_classes=num_classes,
        use_class_weight=use_class_weight and split == 'train',
        use_class_sampling=use_class_sampling and split == 'train',
        class_weight_power=class_weight_power,
        class_sampling_power=class_sampling_power,
    )


model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=3,
        num_person=1,
        base_channels=32,
        num_prototype=200,
        view_num=0,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02),
    ),
    cls_head=dict(
        type='SimpleHead',
        joint_cfg=graph,
        num_classes=num_classes,
        in_channels=128,
        weight=0.2,
    ),
    view_loss_weight=0.0,
    test_cfg=dict(feat_ext=False),
)

train_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=500),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=500, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=500, num_clips=10),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['subject', 'patient_id', 'cohort', 'transition_id', 'frame_dir', 'total_frames']),
    dict(type='ToTensor', keys=['keypoint']),
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train_eval_dataloader=dict(videos_per_gpu=1),
    test_eval_dataloader=dict(videos_per_gpu=1),
    train=_dataset_cfg('train', fold),
    train_eval=_dataset_cfg('train', fold, pipeline=val_pipeline),
    val=_dataset_cfg('test', fold),
    test_eval=_dataset_cfg('test', fold, pipeline=val_pipeline),
    test=_dataset_cfg('test', fold),
)

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-4, by_epoch=True)
total_epochs = 10
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=1,
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
)
train_evaluation = dict(
    interval=1,
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
    save_best=None,
)
test_evaluation = dict(
    interval=1,
    metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
    save_best=None,
)
use_val = False
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
