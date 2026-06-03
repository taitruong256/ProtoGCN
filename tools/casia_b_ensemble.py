"""Ensemble script for CASIA-B 6-modality ensemble"""
from mmcv import load, dump
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from protogcn.smp import comb, top1, load_label

# Paths for CASIA-B predictions
joint_path = '../work_dirs/casia_b/j/best_pred.pkl'
bone_path = '../work_dirs/casia_b/b/best_pred.pkl'
kbone_path = '../work_dirs/casia_b/k/best_pred.pkl'
joint_motion_path = '../work_dirs/casia_b/jm/best_pred.pkl'
bone_motion_path = '../work_dirs/casia_b/bm/best_pred.pkl'
kbone_motion_path = '../work_dirs/casia_b/km/best_pred.pkl'

# Load predictions
print("Loading predictions...")
try:
    joint = load(joint_path)
    bone = load(bone_path)
    kbone = load(kbone_path)
    joint_motion = load(joint_motion_path)
    bone_motion = load(bone_motion_path)
    kbone_motion = load(kbone_motion_path)
    print("✓ All predictions loaded successfully")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Make sure to run inference for all modalities first:")
    print("  bash tools/dist_test.sh configs/casia_b/j.py <checkpoint> 1 --out work_dirs/casia_b/j/best_pred.pkl")
    print("  bash tools/dist_test.sh configs/casia_b/b.py <checkpoint> 1 --out work_dirs/casia_b/b/best_pred.pkl")
    print("  ... (repeat for k, jm, bm, km)")
    sys.exit(1)

# Load CASIA-B labels - assuming valid set
# Note: Update paths if needed
label_path = 'data/casia-b/casia-b_labels.pkl'
if not os.path.exists(label_path):
    print(f"Warning: Label file {label_path} not found")
    print("Skipping label-based evaluation")
    label = None
else:
    label = load(label_path)

print("\n" + "="*60)
print("CASIA-B 6-Modality Ensemble Results")
print("="*60)

# Test different ensemble strategies
results = {}

# 2-stream: J+B
if label is not None:
    print('\n[2-Stream] J+B')
    fused = comb([joint, bone], [1, 1])
    acc = top1(fused, label)
    results['2-stream'] = acc
    print(f'Top-1 Accuracy: {acc:.4f}')
else:
    print('\n[2-Stream] J+B - Label file missing, skipping evaluation')

# 4-stream: J+B+JM+BM
if label is not None:
    print('\n[4-Stream] J+B+JM+BM')
    fused = comb([joint, bone, joint_motion, bone_motion], [2, 2, 1, 1])
    acc = top1(fused, label)
    results['4-stream'] = acc
    print(f'Top-1 Accuracy: {acc:.4f}')
else:
    print('\n[4-Stream] J+B+JM+BM - Label file missing, skipping evaluation')

# 6-stream: J+B+K+JM+BM+KM (the best performing ensemble)
if label is not None:
    print('\n[6-Stream] J+B+K+JM+BM+KM (RECOMMENDED)')
    fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion], [2, 2, 2, 1, 1, 1])
    acc = top1(fused, label)
    results['6-stream'] = acc
    print(f'Top-1 Accuracy: {acc:.4f} ⭐ Best Ensemble')
else:
    print('\n[6-Stream] J+B+K+JM+BM+KM - Label file missing, skipping evaluation')

# Save ensemble results
if results:
    print("\n" + "="*60)
    print("Summary:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")
    print("="*60)

# Optionally save the fused predictions
# dump(fused, '../work_dirs/casia_b/ensemble/6stream_pred.pkl')
print("\nNote: Labels file is required for accuracy evaluation.")
print("Ensure you have the proper label file in the path specified.")
