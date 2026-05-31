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
                 min_frames=1,
                 **kwargs):
        self.gallery_conditions = set(gallery_conditions)
        self.gallery_sequences = set(gallery_sequences)
        self.probe_conditions = set(probe_conditions)
        self.probe_nm_sequences = set(probe_nm_sequences)
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
        if condition in self.probe_conditions:
            return 'probe'
        return 'ignore'

    @staticmethod
    def _to_feature(result):
        feat = np.asarray(result, dtype=np.float32)
        feat = feat.reshape(-1, feat.shape[-3]) if feat.ndim >= 3 else feat.reshape(1, -1)
        feat = feat.mean(axis=0)
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
        allowed_metrics = ['gait_rank1', 'gait_contrastive_loss']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        features = np.stack([self._to_feature(result) for result in results])
        labels = np.array([ann['label'] for ann in self.video_infos])
        roles = np.array([ann.get('gait_role', 'probe') for ann in self.video_infos])

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
            gallery_features = features[gallery_mask]
            gallery_labels = labels[gallery_mask]
            probe_features = features[probe_mask]
            probe_labels = labels[probe_mask]
            probe_conditions = np.array([ann.get('condition', '') for ann in self.video_infos])[probe_mask]

            gallery_templates = []
            gallery_template_labels = []
            for label in sorted(set(gallery_labels)):
                label_mask = gallery_labels == label
                template = gallery_features[label_mask].mean(axis=0)
                norm = np.linalg.norm(template)
                if norm > 0:
                    template = template / norm
                gallery_templates.append(template)
                gallery_template_labels.append(label)
            gallery_templates = np.stack(gallery_templates)
            gallery_template_labels = np.asarray(gallery_template_labels)

            distances = 1 - np.matmul(probe_features, gallery_templates.T)
            pred_labels = gallery_template_labels[np.argmin(distances, axis=1)]
            correct = pred_labels == probe_labels
            eval_results['gait_rank1'] = float(correct.mean())
            print_log(f'\ngait_rank1\t{eval_results["gait_rank1"]:.4f}', logger=logger)

            for condition in ('bg', 'cl', 'nm'):
                condition_mask = probe_conditions == condition
                if not np.any(condition_mask):
                    continue
                key = f'gait_rank1_{condition}'
                eval_results[key] = float(correct[condition_mask].mean())
                print_log(f'\n{key}\t{eval_results[key]:.4f}', logger=logger)

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
