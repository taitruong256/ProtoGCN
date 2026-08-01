"""Train the four-branch CARE-PD model and merge 6-fold predictions.

The model contains four independent branches. Validation/test predictions are
the mean of their probabilities, so ``save_best='f1_score'`` selects one
same-epoch checkpoint using the four-branch ensemble metric.
"""

import argparse
import csv
import glob
import json
import os
import os.path as osp
import subprocess
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import Config

from protogcn.datasets import build_dataset


DEFAULT_CONFIG = 'configs/care_pd/pd_gam_4stage_multibranch.py'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=DEFAULT_CONFIG)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1, 7)))
    parser.add_argument(
        '--output-dir', default='work_dirs/care_pd/pd_gam_4branch_6fold',
        help='directory for per-fold and merged ensemble predictions')
    parser.add_argument(
        '--skip-train', action='store_true',
        help='predict/aggregate using existing best ensemble checkpoints')
    parser.add_argument(
        '--aggregate-only', action='store_true',
        help='only rebuild merged metrics from saved fold prediction CSV files')
    parser.add_argument('--save-json', default=None,
                        help='optional second path for overall metrics JSON')
    return parser.parse_args()


def run_command(cmd, env):
    print('\n$ ' + ' '.join(cmd), flush=True)
    process = subprocess.run(cmd, env=env, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f'Command failed ({process.returncode}): {" ".join(cmd)}')


def _safe_divide(numerator, denominator):
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator != 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def quadratic_weighted_kappa(confusion_matrix):
    confusion = np.asarray(confusion_matrix, dtype=np.float64)
    count = confusion.sum()
    if count == 0:
        return float('nan')
    num_classes = confusion.shape[0]
    indices = np.arange(num_classes, dtype=np.float64)
    weights = (indices[:, None] - indices[None, :]) ** 2
    if num_classes > 1:
        weights /= (num_classes - 1) ** 2
    expected = np.outer(confusion.sum(1), confusion.sum(0)) / count
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0
    return float(1 - (weights * confusion).sum() / denominator)


def compute_metrics(scores, labels, num_classes):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape != (len(labels), num_classes):
        raise ValueError(
            f'Expected scores ({len(labels)}, {num_classes}), got {scores.shape}')
    if len(labels) == 0:
        raise ValueError('Cannot evaluate an empty prediction set')
    predictions = scores.argmax(axis=1).astype(np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(labels, predictions):
        if not 0 <= target < num_classes:
            raise ValueError(f'Label {target} is outside [0, {num_classes})')
        cm[target, prediction] += 1

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm.sum(0).astype(np.float64)
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return OrderedDict([
        ('num_sequences', int(len(labels))),
        ('accuracy', float((predictions == labels).mean())),
        ('balanced_accuracy', float(recall.mean())),
        ('precision', float(precision.mean())),
        ('recall', float(recall.mean())),
        ('f1_score', float(f1.mean())),
        ('mae', float(np.abs(predictions - labels).mean())),
        ('quadratic_weighted_kappa', quadratic_weighted_kappa(cm)),
        ('confusion_matrix', cm.tolist()),
        ('per_class', {
            str(class_id): {
                'support': int(support[class_id]),
                'precision': float(precision[class_id]),
                'recall': float(recall[class_id]),
                'f1': float(f1[class_id]),
            }
            for class_id in range(num_classes)
        }),
    ])


def load_fold_config(config_path, fold_id):
    old_fold = os.environ.get('CARE_PD_FOLD')
    os.environ['CARE_PD_FOLD'] = str(fold_id)
    try:
        return Config.fromfile(config_path)
    finally:
        if old_fold is None:
            os.environ.pop('CARE_PD_FOLD', None)
        else:
            os.environ['CARE_PD_FOLD'] = old_fold


def build_test_data(config_path, fold_id):
    cfg = load_fold_config(config_path, fold_id)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    labels = np.asarray(
        [item['label'] for item in dataset.video_infos], dtype=np.int64)
    keys = [
        (str(item.get('participant', '')), str(item.get('sequence', '')))
        for item in dataset.video_infos
    ]
    return cfg, labels, keys


def find_best_checkpoint(work_dir):
    latest_path = osp.join(work_dir, 'latest.pth')
    if osp.exists(latest_path):
        try:
            checkpoint = torch.load(
                latest_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest_path, map_location='cpu')
        best_path = checkpoint.get('meta', {}).get('hook_msgs', {}).get('best_ckpt')
        if best_path:
            if not osp.isabs(best_path):
                best_path = osp.join(work_dir, osp.basename(best_path))
            if osp.exists(best_path):
                return osp.abspath(best_path)
    candidates = glob.glob(osp.join(work_dir, 'best*.pth'))
    if not candidates:
        raise FileNotFoundError(
            f'No best ensemble checkpoint found in {osp.abspath(work_dir)}')
    return osp.abspath(max(candidates, key=osp.getmtime))


def write_predictions(path, fold_id, keys, labels, scores):
    fields = ['fold', 'participant_id', 'sequence_id', 'target', 'prediction']
    fields += [f'probability_{index}' for index in range(scores.shape[1])]
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (participant, sequence), target, score in zip(keys, labels, scores):
            writer.writerow({
                'fold': fold_id,
                'participant_id': participant,
                'sequence_id': sequence,
                'target': int(target),
                'prediction': int(score.argmax()),
                **{f'probability_{index}': float(value)
                   for index, value in enumerate(score)},
            })


def read_predictions(path, num_classes):
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['fold'] = int(row['fold'])
        row['target'] = int(row['target'])
        row['prediction'] = int(row['prediction'])
        for index in range(num_classes):
            row[f'probability_{index}'] = float(row[f'probability_{index}'])
    return rows


def print_metrics(title, metrics):
    print(f'\n[{title}]')
    for key in ('num_sequences', 'accuracy', 'balanced_accuracy', 'precision',
                'recall', 'f1_score', 'mae', 'quadratic_weighted_kappa'):
        value = metrics[key]
        print(f'{key}: {value}' if isinstance(value, int) else f'{key}: {value:.4f}')
    print('confusion_matrix:')
    print(np.asarray(metrics['confusion_matrix']))


def train_and_predict_fold(config_path, fold_id, args, repo_root):
    env = os.environ.copy()
    env['CARE_PD_FOLD'] = str(fold_id)
    env['PYTHONPATH'] = repo_root + (
        ':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cfg, labels, keys = build_test_data(config_path, fold_id)
    fold_dir = osp.join(args.output_dir, f'fold_{fold_id}')
    mmcv.mkdir_or_exist(fold_dir)

    if not args.skip_train:
        run_command([
            'bash', 'tools/dist_train.sh', config_path, str(args.gpus),
            '--validate'
        ], env)

    checkpoint_path = find_best_checkpoint(cfg.work_dir)
    prediction_path = osp.join(fold_dir, 'ensemble_best_pred.pkl')
    run_command([
        'bash', 'tools/dist_test.sh', config_path, checkpoint_path,
        str(args.gpus), '--out', prediction_path, '--eval',
        'accuracy', 'f1_score', 'precision', 'recall'
    ], env)
    scores = np.asarray(mmcv.load(prediction_path), dtype=np.float32)
    num_classes = int(cfg.model['cls_head']['num_classes'])
    if scores.shape != (len(labels), num_classes):
        raise ValueError(
            f'Fold {fold_id}: expected {(len(labels), num_classes)}, '
            f'got {scores.shape}')

    write_predictions(
        osp.join(fold_dir, 'predictions.csv'), fold_id, keys, labels, scores)
    metrics = compute_metrics(scores, labels, num_classes)
    metrics['best_checkpoint'] = checkpoint_path
    with open(osp.join(fold_dir, 'metrics.json'), 'w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2)
    print_metrics(f'Fold {fold_id} ensemble', metrics)
    return num_classes


def aggregate_predictions(output_dir, folds, num_classes):
    rows = []
    for fold_id in folds:
        path = osp.join(output_dir, f'fold_{fold_id}', 'predictions.csv')
        if not osp.exists(path):
            raise FileNotFoundError(f'Missing fold predictions: {path}')
        rows.extend(read_predictions(path, num_classes))
    keys = [(row['participant_id'], row['sequence_id']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate held-out sequences found across folds')
    rows.sort(key=lambda row: (row['participant_id'], row['sequence_id']))

    labels = np.asarray([row['target'] for row in rows], dtype=np.int64)
    scores = np.asarray([
        [row[f'probability_{index}'] for index in range(num_classes)]
        for row in rows
    ], dtype=np.float32)
    metrics = compute_metrics(scores, labels, num_classes)
    metrics['num_participants'] = len({row['participant_id'] for row in rows})
    metrics['folds'] = list(folds)

    with open(osp.join(output_dir, 'all_folds_predictions.csv'), 'w',
              newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    mmcv.dump([score for score in scores],
              osp.join(output_dir, 'all_folds_ensemble.pkl'))
    with open(osp.join(output_dir, 'overall_metrics.json'), 'w',
              encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2)
    print_metrics('Merged out-of-fold ensemble', metrics)
    return metrics


def main():
    args = parse_args()
    if not args.folds or any(fold not in range(1, 7) for fold in args.folds):
        raise ValueError('--folds only accepts values from 1 to 6')
    if len(set(args.folds)) != len(args.folds):
        raise ValueError('--folds contains duplicate fold IDs')
    if args.aggregate_only and args.skip_train:
        raise ValueError('--aggregate-only already implies --skip-train')

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    os.chdir(repo_root)
    config_path = osp.abspath(args.config)
    args.output_dir = osp.abspath(args.output_dir)
    mmcv.mkdir_or_exist(args.output_dir)
    first_cfg = load_fold_config(config_path, args.folds[0])
    num_classes = int(first_cfg.model['cls_head']['num_classes'])

    if not args.aggregate_only:
        for fold_id in args.folds:
            fold_classes = train_and_predict_fold(
                config_path, fold_id, args, repo_root)
            if fold_classes != num_classes:
                raise ValueError('Number of classes changed between folds')
    metrics = aggregate_predictions(args.output_dir, args.folds, num_classes)

    if args.save_json:
        save_path = osp.abspath(args.save_json)
        mmcv.mkdir_or_exist(osp.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, indent=2)
        print(f'Copied overall metrics to: {save_path}')


if __name__ == '__main__':
    main()
