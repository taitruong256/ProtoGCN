import copy
import csv
import os.path as osp
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


COCO_KEYPOINTS = (
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle')


@DATASETS.register_module()
class CasiaBGaitDataset(BaseDataset):
    """CASIA-B skeleton gait dataset loaded from per-frame COCO CSV files.

    The CSV is expected to contain one row per frame. The sequence metadata is
    parsed from paths like ``001-bg-01-000/000001.jpg``: subject, condition,
    sequence id and camera view.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 gallery_conditions=('nm', ),
                 gallery_sequences=('01', '02', '03', '04'),
                 probe_nm_sequences=('05', '06'),
                 probe_conditions=('bg', 'cl'),
                 probe_bgcl_sequences=('01', '02'),
                 min_frames=1,
                 **kwargs):
        self.gallery_conditions = set(gallery_conditions)
        self.gallery_sequences = set(gallery_sequences)
        self.probe_conditions = set(probe_conditions)
        self.probe_nm_sequences = set(probe_nm_sequences)
        self.probe_bgcl_sequences = set(probe_bgcl_sequences)
        self.min_frames = min_frames
        super().__init__(ann_file, pipeline, start_index=0, modality='Pose', **kwargs)

        logger = get_root_logger()
        logger.info(f'{len(self)} CASIA-B gait sequences loaded')

    def load_annotations(self):
        assert self.ann_file.endswith('.csv')
        grouped = defaultdict(list)

        with open(self.ann_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row['image_name']
                seq_name = osp.normpath(image_name).split(osp.sep)[0]
                if seq_name == '.':
                    seq_name = osp.normpath(image_name).split(osp.sep)[1]
                subject, condition, sequence, view = seq_name.split('-')
                frame_name = osp.splitext(osp.basename(image_name))[0]
                frame_id = int(frame_name)

                keypoint = []
                keypoint_score = []
                for name in COCO_KEYPOINTS:
                    keypoint.append([float(row[f'{name}_x']), float(row[f'{name}_y'])])
                    keypoint_score.append(float(row[f'{name}_conf']))

                grouped[seq_name].append((
                    frame_id, np.array(keypoint, dtype=np.float32),
                    np.array(keypoint_score, dtype=np.float32),
                    subject, condition, sequence, view))

        data = []
        for seq_name, frames in sorted(grouped.items()):
            if len(frames) < self.min_frames:
                continue
            frames = sorted(frames, key=lambda x: x[0])
            _, keypoints, scores, subject, condition, sequence, view = zip(*frames)
            subject = subject[0]
            condition = condition[0]
            sequence = sequence[0]
            view = view[0]
            gait_role = self._get_gait_role(condition, sequence)
            if self.test_mode and gait_role == 'ignore':
                continue

            item = dict(
                frame_dir=seq_name,
                total_frames=len(frames),
                label=int(subject) - 1,
                subject=subject,
                condition=condition,
                sequence=sequence,
                view=view,
                gait_role=gait_role,
                keypoint=np.stack(keypoints, axis=0)[None],
                keypoint_score=np.stack(scores, axis=0)[None])
            data.append(item)
        return data

    def _get_gait_role(self, condition, sequence):
        if condition in self.gallery_conditions and sequence in self.gallery_sequences:
            return 'gallery'
        if condition == 'nm' and sequence in self.probe_nm_sequences:
            return 'probe'
        if (condition in self.probe_conditions
                and sequence in self.probe_bgcl_sequences):
            return 'probe'
        return 'ignore'

    @staticmethod
    def _view_to_index(view, num_views=11):
        if isinstance(view, (list, tuple)) and len(view) > 0:
            view = view[0]
        if hasattr(view, 'item'):
            try:
                view = view.item()
            except Exception:
                pass
        if isinstance(view, str):
            view = view.strip()
            if view.isdigit():
                view = int(view)
            else:
                view = int(float(view))
        view = int(view)
        if 0 <= view < num_views:
            return view
        if 0 <= view <= 180 and view % 18 == 0:
            return view // 18
        raise ValueError(f'Unsupported view value: {view}')

    @staticmethod
    def _to_feature(result):
        # Match the ensemble evaluator: flatten one prediction and then
        # apply L2 normalization before cosine-distance matching.
        feat = np.asarray(result, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    @staticmethod
    def _contrastive_loss(features, labels, temperature=0.07):
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        n = len(labels)
        if n <= 1:
            return 0.0

        sim = np.matmul(features, features.T) / temperature
        sim = sim - np.max(sim, axis=1, keepdims=True)
        exp_sim = np.exp(sim) * (1 - np.eye(n, dtype=np.float32))
        log_prob = sim - np.log(exp_sim.sum(axis=1, keepdims=True) + 1e-12)
        pos_mask = (labels[:, None] == labels[None, :]) & (~np.eye(n, dtype=bool))
        pos_count = pos_mask.sum(axis=1)
        valid = pos_count > 0
        if not np.any(valid):
            return 0.0
        loss = -(log_prob * pos_mask).sum(axis=1)[valid] / pos_count[valid]
        return float(loss.mean())

    @staticmethod
    def _fastposegait_rank1(features, labels, conditions, sequences, views, roles):
        """Evaluate Rank-1 with the same protocol as the ensemble script.

        Gallery sequences are averaged into one normalized template per
        identity. Probe features are matched to those templates using cosine
        distance (``1 - cosine_similarity``), exactly as in
        ``tools/casia_b_ensemble.py``.
        """
        gallery_mask = roles == 'gallery'
        probe_mask = roles == 'probe'
        if not np.any(gallery_mask):
            raise ValueError('CASIA-B gait evaluation requires gallery sequences.')

        if not np.any(probe_mask):
            raise ValueError('CASIA-B gait evaluation requires probe sequences.')

        gallery_features = features[gallery_mask]
        gallery_labels = labels[gallery_mask]
        probe_features = features[probe_mask]
        probe_labels = labels[probe_mask]
        probe_conditions = conditions[probe_mask]

        gallery_templates = []
        gallery_template_labels = []
        for label in sorted(set(gallery_labels.tolist())):
            label_mask = gallery_labels == label
            template = gallery_features[label_mask].mean(axis=0)
            norm = np.linalg.norm(template)
            if norm > 0:
                template = template / norm
            gallery_templates.append(template)
            gallery_template_labels.append(label)

        gallery_templates = np.stack(gallery_templates, axis=0)
        gallery_template_labels = np.asarray(gallery_template_labels)

        distances = 1 - np.matmul(probe_features, gallery_templates.T)
        pred_labels = gallery_template_labels[np.argmin(distances, axis=1)]
        correct = pred_labels == probe_labels

        results = OrderedDict()
        results['gait_rank1'] = float(correct.mean())
        for condition in ('bg', 'cl', 'nm'):
            condition_mask = probe_conditions == condition
            if np.any(condition_mask):
                results[f'gait_rank1_{condition}'] = float(
                    correct[condition_mask].mean())
        return results

    def evaluate(self,
                 results,
                 metrics='gait_rank1',
                 metric_options=dict(gait_contrastive_loss=dict(temperature=0.07)),
                 logger=None,
                 **deprecated_kwargs):
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['gait_rank1', 'gait_contrastive_loss', 'view_acc']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        view_scores = None
        if len(results) > 0 and isinstance(results[0], (tuple, list)):
            features = np.stack([self._to_feature(result[0]) for result in results])
            if len(results[0]) > 1:
                view_scores = np.stack([
                    np.asarray(result[1], dtype=np.float32) for result in results
                ])
        else:
            features = np.stack([self._to_feature(result) for result in results])
        labels = np.array([ann['label'] for ann in self.video_infos])
        roles = np.array([ann.get('gait_role', 'probe') for ann in self.video_infos])
        conditions = np.array([ann.get('condition', '') for ann in self.video_infos])
        sequences = np.array([str(ann.get('sequence', '')) for ann in self.video_infos])
        view_labels = np.array([
            self._view_to_index(ann.get('view', 0)) for ann in self.video_infos
        ])

        gallery_mask = roles == 'gallery'
        probe_mask = roles == 'probe'
        if not np.any(gallery_mask):
            raise ValueError('CASIA-B gait evaluation requires at least one gallery sequence.')
        if not np.any(probe_mask):
            raise ValueError('CASIA-B gait evaluation requires at least one probe sequence.')

        eval_results = OrderedDict()
        if 'gait_rank1' in metrics:
            msg = '\nEvaluating gait_rank1 ...' if logger is None else 'Evaluating gait_rank1 ...'
            print_log(msg, logger=logger)
            rank1_results = self._fastposegait_rank1(
                features, labels, conditions, sequences, view_labels, roles)
            eval_results.update(rank1_results)
            for key, value in rank1_results.items():
                print_log(f'\n{key}\t{value:.4f}', logger=logger)

        if 'gait_contrastive_loss' in metrics:
            msg = '\nEvaluating gait_contrastive_loss ...' if logger is None else 'Evaluating gait_contrastive_loss ...'
            print_log(msg, logger=logger)
            temperature = metric_options.setdefault(
                'gait_contrastive_loss', {}).setdefault('temperature', 0.07)
            eval_results['gait_contrastive_loss'] = self._contrastive_loss(
                features, labels, temperature=temperature)
            print_log(
                f'\ngait_contrastive_loss\t{eval_results["gait_contrastive_loss"]:.4f}',
                logger=logger)

        if 'view_acc' in metrics:
            if view_scores is None:
                raise ValueError(
                    'view_acc was requested, but model outputs did not include view scores. '
                    'Enable `return_view_score=True` in test_cfg.'
                )
            msg = '\nEvaluating view_acc ...' if logger is None else 'Evaluating view_acc ...'
            print_log(msg, logger=logger)
            pred_view = np.argmax(view_scores, axis=1)
            eval_results['view_acc'] = float((pred_view == view_labels).mean())
            print_log(f'\nview_acc\t{eval_results["view_acc"]:.4f}', logger=logger)

        return eval_results

    @staticmethod
    def dump_results(results, out):
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['test_mode'] = self.test_mode
        results['idx'] = idx
        return self.pipeline(results)
