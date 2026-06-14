"""Ensemble script for CASIA-B 6-modality ensemble"""
import argparse
from mmcv import load
import csv
import os
import pickle
import sys
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protogcn.smp import comb

# Paths for CASIA-B predictions
joint_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/j_new/best_pred.pkl'
bone_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/b_new/best_pred.pkl'
kbone_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/k_new/best_pred.pkl'
joint_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/jm_new/best_pred.pkl'
bone_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/bm_new/best_pred.pkl'
kbone_motion_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/km_new/best_pred.pkl'
angle_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/a_new/best_pred.pkl'
relative_path = '/home/HardDisk/Tai/ProtoGCN/work_dirs/casia_b/r_new/best_pred.pkl'


def _seq_name_from_image_name(image_name):
    seq_name = os.path.normpath(image_name).split(os.sep)[0]
    if seq_name == '.':
        seq_name = os.path.normpath(image_name).split(os.sep)[1]
    return seq_name


def build_casia_b_label_pkl(csv_path, pkl_path):
    """Build a list[int] label file from a CASIA-B CSV annotation file."""
    labels_by_seq = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = _seq_name_from_image_name(row['image_name'])
            if seq_name not in labels_by_seq:
                subject = seq_name.split('-')[0]
                labels_by_seq[seq_name] = int(subject) - 1

    labels = [labels_by_seq[seq_name] for seq_name in sorted(labels_by_seq)]
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(labels, f)
    return labels


def _gait_role(condition, sequence, gallery_conditions=('nm',),
               gallery_sequences=('01', '02', '03', '04'),
               probe_nm_sequences=('05', '06'),
               probe_conditions=('bg', 'cl')):
    if condition in set(gallery_conditions) and sequence in set(gallery_sequences):
        return 'gallery'
    if condition == 'nm' and sequence in set(probe_nm_sequences):
        return 'probe'
    if condition in set(probe_conditions):
        return 'probe'
    return 'ignore'


def load_seq_infos(csv_path):
    seq_infos = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_name = _seq_name_from_image_name(row['image_name'])
            if seq_name in seq_infos:
                continue
            subject, condition, sequence, view = seq_name.split('-')
            seq_infos[seq_name] = dict(
                frame_dir=seq_name,
                label=int(subject) - 1,
                subject=subject,
                condition=condition,
                sequence=sequence,
                view=view,
                gait_role=_gait_role(condition, sequence),
            )
    return [seq_infos[seq_name] for seq_name in sorted(seq_infos)]


def _normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def gait_rank1(features, seq_infos):
    features = np.asarray([_normalize_feature(x) for x in features], dtype=np.float32)
    labels = np.asarray([ann['label'] for ann in seq_infos])
    roles = np.asarray([ann.get('gait_role', 'probe') for ann in seq_infos])

    gallery_mask = roles == 'gallery'
    probe_mask = roles == 'probe'
    if not np.any(gallery_mask):
        raise ValueError('CASIA-B gait evaluation requires at least one gallery sequence.')
    if not np.any(probe_mask):
        raise ValueError('CASIA-B gait evaluation requires at least one probe sequence.')

    gallery_features = features[gallery_mask]
    gallery_labels = labels[gallery_mask]
    probe_features = features[probe_mask]
    probe_labels = labels[probe_mask]
    probe_conditions = np.asarray([ann.get('condition', '') for ann in seq_infos])[probe_mask]

    gallery_templates = []
    gallery_template_labels = []
    for label in sorted(set(gallery_labels.tolist())):
        label_mask = gallery_labels == label
        template = gallery_features[label_mask].mean(axis=0)
        template = _normalize_feature(template)
        gallery_templates.append(template)
        gallery_template_labels.append(label)

    gallery_templates = np.stack(gallery_templates, axis=0)
    gallery_template_labels = np.asarray(gallery_template_labels)

    distances = 1 - np.matmul(probe_features, gallery_templates.T)
    pred_labels = gallery_template_labels[np.argmin(distances, axis=1)]
    correct = pred_labels == probe_labels

    results = {'gait_rank1': float(correct.mean())}
    for condition in ('bg', 'cl', 'nm'):
        condition_mask = probe_conditions == condition
        if np.any(condition_mask):
            results[f'gait_rank1_{condition}'] = float(correct[condition_mask].mean())
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        choices=['valid', 'test'],
        default='valid',
        help='CASIA-B split used for labels and evaluation')
    return parser.parse_args()

args = parse_args()
csv_path = f'data/casia-b/casia-b_pose_{args.split}.csv'

# Load predictions
print("Loading predictions...")
try:
    joint = load(joint_path)
    bone = load(bone_path)
    kbone = load(kbone_path)
    joint_motion = load(joint_motion_path)
    bone_motion = load(bone_motion_path)
    kbone_motion = load(kbone_motion_path)
    angle = load(angle_path)
    relative = load(relative_path)
    print("All predictions loaded successfully")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure to run inference for all modalities first:")
    print("  bash tools/dist_test.sh configs/casia_b/j.py <checkpoint> 1 --out work_dirs/casia_b/j/best_pred.pkl")
    print("  bash tools/dist_test.sh configs/casia_b/b.py <checkpoint> 1 --out work_dirs/casia_b/b/best_pred.pkl")
    print("  ... (repeat for k, jm, bm, km)")
    sys.exit(1)

# Load CASIA-B labels from the selected CSV and cache them as a pkl file.
label_path = f'data/casia-b/casia-b_labels_{args.split}.pkl'
if not os.path.exists(label_path):
    if not os.path.exists(csv_path):
        print(f"Warning: Label CSV {csv_path} not found")
        print("Skipping label-based evaluation")
        label = None
    else:
        print(f"Label file {label_path} not found, generating it from {csv_path}")
        label = build_casia_b_label_pkl(csv_path, label_path)
        print(f"Saved {label_path}")
else:
    label = load(label_path)
    if label and isinstance(label[0], dict):
        label = [x['label'] for x in label]

seq_infos = load_seq_infos(csv_path) if os.path.exists(csv_path) else None
if seq_infos is not None and label is not None and len(seq_infos) != len(label):
    print(f"Warning: label count ({len(label)}) != sequence count ({len(seq_infos)})")

print("\n" + "="*60)
print("CASIA-B 6-Modality Ensemble Results")
print("="*60)

# Test different ensemble strategies
results = {}

def report_stream(name, score):
    metrics = gait_rank1(score, seq_infos)
    print(f"{name} gait_rank1: {metrics['gait_rank1']:.4f}")
    for key in ('gait_rank1_bg', 'gait_rank1_cl', 'gait_rank1_nm'):
        if key in metrics:
            print(f'  {key}: {metrics[key]:.4f}')
    return metrics

if label is not None:
    print('\n[Single Streams]')
    report_stream('J', joint)
    report_stream('B', bone)
    report_stream('K', kbone)
    report_stream('JM', joint_motion)
    report_stream('BM', bone_motion)
    report_stream('KM', kbone_motion)

# 2-stream: J+B
if label is not None:
    print('\n[2-Stream] J+B')
    fused = comb([joint, bone], [1, 1])
    metrics = report_stream('J+B', fused)
    results['2-stream'] = metrics
else:
    print('\n[2-Stream] J+B - Label file missing, skipping evaluation')

# 2-stream: J + K
if label is not None:
    print('\n[2-Stream] J+K')
    fused = comb([joint, kbone], [1, 1])
    metrics = report_stream('J+K', fused)
    results['2-stream-jk'] = metrics
else:
    print('\n[2-Stream] J+K - Label file missing, skipping evaluation')

# 2-stream: B + K 
if label is not None:
    print('\n[2-Stream] B+K')
    fused = comb([bone, kbone], [1, 1])
    metrics = report_stream('B+K', fused)
    results['2-stream-bk'] = metrics
else:
    print('\n[2-Stream] B+K - Label file missing, skipping evaluation')

# 4-stream: J+B+JM+BM
if label is not None:
    print('\n[4-Stream] J+B+JM+BM')
    fused = comb([joint, bone, joint_motion, bone_motion], [1, 1, 1, 1])
    metrics = report_stream('J+B+JM+BM', fused)
    results['4-stream'] = metrics
else:
    print('\n[4-Stream] J+B+JM+BM - Label file missing, skipping evaluation')

# 4-stream: J + K + JM + KM
if label is not None:
    print('\n[4-Stream] J+K+JM+KM')
    fused = comb([joint, kbone, joint_motion, kbone_motion], [1, 1, 1, 1])
    metrics = report_stream('J+K+JM+KM', fused)
    results['4-stream-jk'] = metrics
else:
    print('\n[4-Stream] J+K+JM+KM - Label file missing, skipping evaluation')

# 4-stream: B + K + BM + KM
if label is not None:
    print('\n[4-Stream] B+K+BM+KM')
    fused = comb([bone, kbone, bone_motion, kbone_motion], [1, 1, 1, 1])
    metrics = report_stream('B+K+BM+KM', fused)
    results['4-stream-bk'] = metrics
else:
    print('\n[4-Stream] B+K+BM+KM - Label file missing, skipping evaluation')

# 6-stream: J+B+K+JM+BM+KM 
if label is not None:
    print('\n[6-Stream] J+B+K+JM+BM+KM')
    fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion], [1, 1, 1, 1, 1, 1])
    metrics = report_stream('J+B+K+JM+BM+KM', fused)
    results['6-stream'] = metrics
else:
    print('\n[6-Stream] J+B+K+JM+BM+KM - Label file missing, skipping evaluation')

# 7-stream: J+B+K+JM+BM+KM+R
if label is not None:
    print('\n[7-Stream] J+B+K+JM+BM+KM+A')
    fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion, relative], [1, 1, 1, 1, 1, 1, 1])
    metrics = report_stream('J+B+K+JM+BM+KM+A', fused)
    results['7-stream'] = metrics
else:
    print('\n[7-Stream] J+B+K+JM+BM+KM+A - Label file missing, skipping evaluation')

# 7-stream: J+B+K+JM+BM+KM+A 
if label is not None:
    print('\n[7-Stream] J+B+K+JM+BM+KM+A')
    fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion, angle], [1, 1, 1, 1, 1, 1, 1])
    metrics = report_stream('J+B+K+JM+BM+KM+A', fused)
    results['7-stream'] = metrics
else:
    print('\n[7-Stream] J+B+K+JM+BM+KM+A - Label file missing, skipping evaluation')

# 8-stream: J+B+K+JM+BM+KM+A+R
if label is not None:
    print('\n[8-Stream] J+B+K+JM+BM+KM+A+R')
    fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion, angle, relative], [1, 1, 1, 1, 1, 1, 1, 1])
    metrics = report_stream('J+B+K+JM+BM+KM+A+R', fused)
    results['8-stream'] = metrics
else:
    print('\n[8-Stream] J+B+K+JM+BM+KM+A+R - Label file missing, skipping evaluation')


# Save ensemble results
if results:
    print("\n" + "="*60)
    print("Summary:")
    for key, val in results.items():
        print(f"  {key}: {val['gait_rank1']:.4f}")
        for cond_key in ('gait_rank1_bg', 'gait_rank1_cl', 'gait_rank1_nm'):
            if cond_key in val:
                print(f"    {cond_key}: {val[cond_key]:.4f}")
    print("="*60)

# Optionally save the fused predictions
# dump(fused, '../work_dirs/casia_b/ensemble/7stream_pred.pkl')