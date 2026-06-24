import argparse
import json
import os
import os.path as osp
import subprocess
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import Config

from protogcn.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate CARE-PD with 6-fold CV')
    parser.add_argument(
        'config',
        default='configs/care_pd/pd_gam_6fold.py',
        nargs='?',
        help='config path')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus for dist_train.sh/dist_test.sh')
    parser.add_argument('--fold', type=int, choices=range(1, 7), default=None,
                        help='train/evaluate one fold only; default runs all 6 folds')
    parser.add_argument('--skip-train', action='store_true', help='skip training and only evaluate existing checkpoints')
    parser.add_argument('--save-json', default=None, help='path to save final fold metrics json')
    return parser.parse_args()


def run_command(cmd, env):
    process = subprocess.run(cmd, env=env, check=False)
    if process.returncode != 0:
        raise RuntimeError(f'Command failed ({process.returncode}): {" ".join(cmd)}')


def _safe_divide(numerator, denominator):
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator != 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def compute_metrics(outputs, gt_labels, num_classes):
    scores = np.asarray(outputs, dtype=np.float32)
    pred_labels = np.argmax(scores, axis=1).astype(np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(gt_labels, pred_labels):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            cm[gt, pred] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision_per_class = _safe_divide(tp, tp + fp)
    recall_per_class = _safe_divide(tp, tp + fn)
    f1_per_class = _safe_divide(2 * precision_per_class * recall_per_class,
                                precision_per_class + recall_per_class)

    metrics = OrderedDict()
    metrics['accuracy'] = float((pred_labels == gt_labels).mean())
    metrics['precision'] = float(precision_per_class.mean())
    metrics['recall'] = float(recall_per_class.mean())
    metrics['f1_score'] = float(f1_per_class.mean())
    metrics['confusion_matrix'] = cm
    return metrics


def build_test_labels(config_path, fold_id):
    os.environ['CARE_PD_FOLD'] = str(fold_id)
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    gt_labels = [ann['label'] for ann in dataset.video_infos]
    return cfg, gt_labels


def main():
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    config_abs = osp.abspath(args.config)

    all_fold_metrics = OrderedDict()
    scalar_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

    fold_ids = [args.fold] if args.fold is not None else range(1, 7)

    for fold_id in fold_ids:
        env = os.environ.copy()
        env['CARE_PD_FOLD'] = str(fold_id)
        env['PYTHONPATH'] = repo_root + (':' + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')

        cfg, gt_labels = build_test_labels(config_abs, fold_id)
        work_dir = cfg.work_dir
        mmcv.mkdir_or_exist(work_dir)
        test_output = osp.join(work_dir, f'fold_{fold_id}_result.pkl')

        if not args.skip_train:
            train_cmd = ['bash', 'tools/dist_train.sh', config_abs, str(args.gpus), '--validate', '--test-last']
            run_command(train_cmd, env=env)

        checkpoint_path = osp.join(work_dir, 'latest.pth')
        if not osp.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

        test_cmd = [
            'bash', 'tools/dist_test.sh', config_abs, checkpoint_path, str(args.gpus),
            '--out', test_output, '--eval', 'accuracy', 'f1_score', 'precision', 'recall'
        ]
        run_command(test_cmd, env=env)

        outputs = mmcv.load(test_output)
        num_classes = int(cfg.model['cls_head']['num_classes'])
        fold_metrics = compute_metrics(outputs, gt_labels, num_classes)
        all_fold_metrics[f'fold_{fold_id}'] = {
            'accuracy': fold_metrics['accuracy'],
            'precision': fold_metrics['precision'],
            'recall': fold_metrics['recall'],
            'f1_score': fold_metrics['f1_score'],
            'confusion_matrix': fold_metrics['confusion_matrix'].tolist()
        }

        for key in scalar_metrics:
            scalar_metrics[key].append(fold_metrics[key])

        print(f'\n[Fold {fold_id}]')
        print(f"accuracy={fold_metrics['accuracy']:.4f} "
              f"precision={fold_metrics['precision']:.4f} "
              f"recall={fold_metrics['recall']:.4f} "
              f"f1_score={fold_metrics['f1_score']:.4f}")
        print(f"confusion_matrix=\n{fold_metrics['confusion_matrix']}")

    summary = OrderedDict()
    for key, values in scalar_metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[f'{key}_mean'] = float(arr.mean())
        summary[f'{key}_std'] = float(arr.std(ddof=0))

    all_fold_metrics['summary'] = summary

    summary_title = 'Single-Fold Summary' if args.fold is not None else '6-Fold Summary'
    print(f'\n[{summary_title}]')
    for key, val in summary.items():
        print(f'{key}: {val:.4f}')

    if args.save_json is not None:
        out_path = osp.abspath(args.save_json)
        mmcv.mkdir_or_exist(osp.dirname(out_path))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_fold_metrics, f, indent=2)
        print(f'\nSaved summary json: {out_path}')


if __name__ == '__main__':
    main()
