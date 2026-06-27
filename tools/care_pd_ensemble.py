import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import Config

from protogcn.datasets import build_dataset


DEFAULT_PREDS = [
    'work_dirs/care_pd/pd_gam_j/fold_1/last_pred.pkl',
    '/home/HardDisk/Tai/ProtoGCN/work_dirs/care_pd/pd_gam_b/fold_1/last_pred.pkl',
    '/home/HardDisk/Tai/ProtoGCN/work_dirs/care_pd/pd_gam_k/fold_1/last_pred.pkl',
]


def parse_args():
    parser = argparse.ArgumentParser(description='CARE-PD J/B/K score ensemble')
    parser.add_argument(
        '--preds',
        nargs='+',
        default=DEFAULT_PREDS,
        help='Prediction pkl paths. Defaults to J, B, K fold-1 last_pred.pkl.')
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        default=None,
        help='Fusion weights. Defaults to equal weights.')
    parser.add_argument(
        '--config',
        default='configs/care_pd/pd_gam_6fold_j.py',
        help='CARE-PD config used to rebuild eval labels.')
    parser.add_argument('--fold', type=int, default=1, help='CARE-PD fold id.')
    parser.add_argument('--save-json', default=None, help='Optional path to save ensemble metrics.')
    parser.add_argument('--save-pred', default=None, help='Optional path to save fused scores.')
    return parser.parse_args()


def _safe_divide(numerator, denominator):
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator != 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def compute_metrics(outputs, gt_labels, num_classes):
    scores = np.asarray(outputs, dtype=np.float32)
    pred_labels = np.argmax(scores, axis=1).astype(np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    if len(scores) != len(gt_labels):
        raise ValueError(f'Prediction/label length mismatch: {len(scores)} != {len(gt_labels)}')

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(gt_labels, pred_labels):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            cm[gt, pred] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision_per_class = _safe_divide(tp, tp + fp)
    recall_per_class = _safe_divide(tp, tp + fn)
    f1_per_class = _safe_divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class)

    metrics = OrderedDict()
    metrics['accuracy'] = float((pred_labels == gt_labels).mean())
    metrics['precision'] = float(precision_per_class.mean())
    metrics['recall'] = float(recall_per_class.mean())
    metrics['f1_score'] = float(f1_per_class.mean())
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    metrics['confusion_matrix'] = cm
    return metrics


def load_scores(path):
    if not osp.exists(path):
        raise FileNotFoundError(path)
    scores = np.asarray(mmcv.load(path), dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f'Expected scores with shape (N, C), got {scores.shape}: {path}')
    return scores


def fuse_scores(score_list, weights):
    shapes = [score.shape for score in score_list]
    if len(set(shapes)) != 1:
        raise ValueError(f'Prediction shapes do not match: {shapes}')

    weights = np.asarray(weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError(f'Weights must be non-negative, got {weights.tolist()}')
    if float(weights.sum()) <= 0:
        raise ValueError('At least one weight must be positive.')
    weights = weights / weights.sum()

    fused = np.zeros_like(score_list[0], dtype=np.float32)
    for score, weight in zip(score_list, weights):
        fused += score * weight
    return fused, weights


def build_gt_labels(config_path, fold_id):
    old_fold = os.environ.get('CARE_PD_FOLD')
    os.environ['CARE_PD_FOLD'] = str(fold_id)
    try:
        cfg = Config.fromfile(config_path)
        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    finally:
        if old_fold is None:
            os.environ.pop('CARE_PD_FOLD', None)
        else:
            os.environ['CARE_PD_FOLD'] = old_fold

    gt_labels = [ann['label'] for ann in dataset.video_infos]
    num_classes = int(cfg.model['cls_head']['num_classes'])
    return gt_labels, num_classes


def print_metrics(name, metrics):
    print(f'\n[{name}]')
    print(f"accuracy={metrics['accuracy']:.4f} "
          f"precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} "
          f"f1_score={metrics['f1_score']:.4f}")
    print(f"precision_per_class={np.array2string(metrics['precision_per_class'], precision=4)}")
    print(f"recall_per_class={np.array2string(metrics['recall_per_class'], precision=4)}")
    print(f"f1_per_class={np.array2string(metrics['f1_per_class'], precision=4)}")
    print(f"confusion_matrix=\n{metrics['confusion_matrix']}")


def metrics_to_jsonable(metrics):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }


def main():
    args = parse_args()
    pred_paths = [osp.abspath(path) for path in args.preds]
    weights = args.weights or [1.0] * len(pred_paths)
    if len(weights) != len(pred_paths):
        raise ValueError(f'Number of weights ({len(weights)}) must match predictions ({len(pred_paths)}).')

    print('Loading CARE-PD labels...')
    gt_labels, num_classes = build_gt_labels(args.config, args.fold)
    print(f'Fold {args.fold}: {len(gt_labels)} samples, {num_classes} classes')

    print('\nLoading predictions...')
    score_list = []
    stream_names = []
    for path in pred_paths:
        score = load_scores(path)
        score_list.append(score)
        stream_names.append(osp.basename(osp.dirname(path)) or osp.basename(path))
        print(f'  {path}: {score.shape}')

    print('\n[Single Streams]')
    all_results = OrderedDict()
    for name, score in zip(stream_names, score_list):
        metrics = compute_metrics(score, gt_labels, num_classes)
        print_metrics(name, metrics)
        all_results[name] = metrics_to_jsonable(metrics)

    fused, norm_weights = fuse_scores(score_list, weights)
    print(f'\nFusion weights: {norm_weights.tolist()}')
    ensemble_metrics = compute_metrics(fused, gt_labels, num_classes)
    print_metrics('Ensemble', ensemble_metrics)
    all_results['ensemble'] = metrics_to_jsonable(ensemble_metrics)
    all_results['weights'] = norm_weights.tolist()
    all_results['pred_paths'] = pred_paths

    if args.save_pred is not None:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.save_pred)))
        mmcv.dump(fused, args.save_pred)
        print(f'\nSaved fused predictions: {args.save_pred}')

    if args.save_json is not None:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.save_json)))
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f'Saved metrics json: {args.save_json}')


if __name__ == '__main__':
    main()
