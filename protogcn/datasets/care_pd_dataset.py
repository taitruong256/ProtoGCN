import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class CarePDDataset(BaseDataset):
    """CARE-PD dataset for UPDRS gait severity classification."""

    def __init__(self,
                 ann_file,
                 pipeline,
                 fold=1,
                 split='train',
                 data_file=None,
                 label_key='UPDRS_GAIT',
                 skeleton_key='skeleton',
                 min_frames=1,
                 **kwargs):
        self.fold = int(fold)
        self.split = split
        self.data_file = data_file
        self.label_key = label_key
        self.skeleton_key = skeleton_key
        self.min_frames = min_frames
        self._warned_pose_fallback = False
        super().__init__(ann_file, pipeline, start_index=0, modality='Pose', **kwargs)

    @staticmethod
    def _normalize_participant_id(participant_id):
        if isinstance(participant_id, str):
            return participant_id.zfill(3)
        return str(participant_id).zfill(3)

    @staticmethod
    def _flatten_fold_source(source_data, fold_id):
        if not isinstance(source_data, dict):
            raise TypeError('data_file content must be a dict.')
        if fold_id in source_data and isinstance(source_data[fold_id], dict):
            fold_data = source_data[fold_id]
            merged = {}
            for split_name in ('train', 'eval'):
                split_data = fold_data.get(split_name, {})
                if isinstance(split_data, dict):
                    merged.update(split_data)
            return merged
        return source_data

    def _extract_skeleton(self, record):
        if self.skeleton_key in record:
            skeleton = np.asarray(record[self.skeleton_key], dtype=np.float32)
            if skeleton.ndim != 3 or skeleton.shape[-1] != 3:
                raise ValueError(f'Invalid skeleton shape: {skeleton.shape}')
            return skeleton

        pose = np.asarray(record.get('pose'))
        if pose.ndim == 2 and pose.shape[1] == 72:
            if not self._warned_pose_fallback:
                print_log(
                    'Skeleton field not found; fallback to reshape pose (T,72)->(T,24,3).',
                    logger=None)
                self._warned_pose_fallback = True
            return pose.reshape(pose.shape[0], 24, 3).astype(np.float32)
        raise KeyError(f'Missing "{self.skeleton_key}" and valid "pose" in record.')

    def load_annotations(self):
        fold_data = mmcv.load(self.ann_file)
        if self.fold not in fold_data:
            raise KeyError(f'Fold {self.fold} not found in {self.ann_file}')
        if self.split not in fold_data[self.fold]:
            raise KeyError(f'Split "{self.split}" not found in fold {self.fold}')

        split_data = fold_data[self.fold][self.split]
        if isinstance(split_data, dict):
            participant_to_sequences = split_data
        else:
            if self.data_file is None:
                raise ValueError('data_file is required when split is participant list.')
            source_data = mmcv.load(self.data_file)
            participant_to_sequences = self._flatten_fold_source(source_data, self.fold)
            participant_to_sequences = {
                self._normalize_participant_id(k): v for k, v in participant_to_sequences.items()
            }
            selected = {
                self._normalize_participant_id(pid) for pid in split_data
            }
            participant_to_sequences = {
                pid: seqs for pid, seqs in participant_to_sequences.items() if pid in selected
            }

        data = []
        for participant_id in sorted(participant_to_sequences.keys()):
            sequences = participant_to_sequences[participant_id]
            if not isinstance(sequences, dict):
                continue
            for sequence_id in sorted(sequences.keys()):
                record = sequences[sequence_id]
                if self.label_key not in record:
                    continue

                skeleton = self._extract_skeleton(record)
                if skeleton.shape[0] < self.min_frames:
                    continue

                item = dict(
                    frame_dir=osp.join(str(participant_id), str(sequence_id)),
                    total_frames=int(skeleton.shape[0]),
                    label=int(record[self.label_key]),
                    participant=str(participant_id),
                    sequence=str(sequence_id),
                    keypoint=skeleton[None])
                data.append(item)
        return data

    @staticmethod
    def _to_scores(results):
        scores = []
        for result in results:
            score = np.asarray(result, dtype=np.float32)
            if score.ndim == 0:
                raise ValueError('Model output must be class scores.')
            if score.ndim > 1:
                score = score.reshape(-1)
            scores.append(score)
        return np.stack(scores, axis=0)

    @staticmethod
    def _safe_divide(numerator, denominator):
        out = np.zeros_like(numerator, dtype=np.float64)
        valid = denominator != 0
        out[valid] = numerator[valid] / denominator[valid]
        return out

    @classmethod
    def _classification_stats(cls, y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        num_classes = labels.size
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for gt, pred in zip(y_true, y_pred):
            cm[label_to_index[gt], label_to_index[pred]] += 1

        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        precision_per_class = cls._safe_divide(tp, tp + fp)
        recall_per_class = cls._safe_divide(tp, tp + fn)
        f1_per_class = cls._safe_divide(2 * precision_per_class * recall_per_class,
                                        precision_per_class + recall_per_class)

        return {
            'accuracy': float((y_true == y_pred).mean()),
            'precision': float(precision_per_class.mean()),
            'recall': float(recall_per_class.mean()),
            'f1_score': float(f1_per_class.mean()),
            'confusion_matrix': cm
        }

    def evaluate(self, results, metrics='accuracy', logger=None, **kwargs):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        if len(results) != len(self):
            raise ValueError(f'Length mismatch: {len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'confusion_matrix']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'Unsupported metric: {metric}')

        y_true = np.asarray([ann['label'] for ann in self.video_infos], dtype=np.int64)
        scores = self._to_scores(results)
        y_pred = np.argmax(scores, axis=1).astype(np.int64)
        stats = self._classification_stats(y_true, y_pred)

        eval_results = OrderedDict()
        for metric in metrics:
            eval_results[metric] = stats[metric]
            if metric != 'confusion_matrix':
                print_log(f'\n{metric}\t{eval_results[metric]:.4f}', logger=logger)
            else:
                print_log(f'\n{metric}\n{eval_results[metric]}', logger=logger)
        return eval_results
