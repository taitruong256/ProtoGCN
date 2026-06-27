import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import Config
from mmcv.fileio.io import file_handlers

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
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Override evaluation metrics, like tools/test.py --eval. Defaults to config.evaluation.metrics.')
    parser.add_argument('--save-json', default=None, help='Optional path to save ensemble metrics.')
    parser.add_argument('--save-pred', default=None, help='Optional path to save fused scores.')
    return parser.parse_args()


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


def load_config_and_eval_cfg(config_path, metrics):
    cfg = Config.fromfile(config_path)
    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if metrics:
        eval_cfg['metrics'] = metrics
    return cfg, eval_cfg


def build_eval_dataset(cfg, fold_id):
    old_fold = os.environ.get('CARE_PD_FOLD')
    os.environ['CARE_PD_FOLD'] = str(fold_id)
    try:
        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    finally:
        if old_fold is None:
            os.environ.pop('CARE_PD_FOLD', None)
        else:
            os.environ['CARE_PD_FOLD'] = old_fold

    return dataset


def evaluate_scores(dataset, scores, metrics):
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) != len(dataset):
        raise ValueError(f'Prediction/dataset length mismatch: {len(scores)} != {len(dataset)}')
    return dataset.evaluate([score for score in scores], **metrics)


def print_metrics(name, metrics):
    print(f'\n[{name}]')
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            print(f'{key}: {float(value):.4f}')
        else:
            print(f'{key}: {value}')


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

    cfg, eval_cfg = load_config_and_eval_cfg(args.config, args.metrics)

    print('Loading CARE-PD labels...')
    dataset = build_eval_dataset(cfg, args.fold)
    gt_labels = [ann['label'] for ann in dataset.video_infos]
    print(f'Fold {args.fold}: {len(dataset)} samples, labels={np.unique(gt_labels).tolist()}')
    print(f'Eval config: {eval_cfg}')

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
        metrics = evaluate_scores(dataset, score, eval_cfg)
        print_metrics(name, metrics)
        all_results[name] = metrics_to_jsonable(metrics)

    fused, norm_weights = fuse_scores(score_list, weights)
    print(f'\nFusion weights: {norm_weights.tolist()}')
    ensemble_metrics = evaluate_scores(dataset, fused, eval_cfg)
    print_metrics('Ensemble', ensemble_metrics)
    all_results['ensemble'] = metrics_to_jsonable(ensemble_metrics)
    all_results['weights'] = norm_weights.tolist()
    all_results['pred_paths'] = pred_paths

    if args.save_pred is not None:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.save_pred)))
        _, suffix = osp.splitext(args.save_pred)
        assert suffix[1:] in file_handlers, 'The output file should be json, pickle/pkl or yaml.'
        dataset.dump_results([score for score in fused], out=args.save_pred)
        print(f'\nSaved fused predictions: {args.save_pred}')

    if args.save_json is not None:
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.save_json)))
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f'Saved metrics json: {args.save_json}')


if __name__ == '__main__':
    main()
