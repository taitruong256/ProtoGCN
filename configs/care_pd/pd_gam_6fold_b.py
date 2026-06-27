import os

modality = 'b'
graph = 'smpl_24'
num_classes = 4

fold_id = int(os.environ.get('CARE_PD_FOLD', 1))
work_dir = f'./work_dirs/care_pd/pd_gam_b/fold_{fold_id}'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=3,
        num_prototype=300,
        gcn_use_view_graph=False,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='SimpleHead', joint_cfg=graph, num_classes=num_classes, in_channels=384, weight=0.2),
    use_view_loss=False,
    test_cfg=dict(feat_ext=False, average_clips='prob'))

dataset_type = 'CarePDDataset'
ann_file = 'data/CARE-PD/folds/UPDRS_Datasets/PD-GaM_6fold_participants_skeleton.pkl'

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=10),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=['participant', 'sequence', 'frame_dir']),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        fold=fold_id,
        split='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        fold=fold_id,
        split='eval',
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        fold=fold_id,
        split='eval',
        pipeline=test_pipeline,
        test_mode=True))

optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 200 
checkpoint_config = dict(interval=1, max_keep_ckpts=1, save_last=True)
evaluation = dict(
    interval=1,
    metrics=['accuracy', 'f1_score', 'precision', 'recall'],
    save_best='f1_score',
    rule='greater')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
