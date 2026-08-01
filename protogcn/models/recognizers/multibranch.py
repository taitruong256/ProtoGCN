import logging

import torch
import torch.nn as nn

from .. import builder
from .base import BaseRecognizer


logger = logging.getLogger(__name__)


@builder.RECOGNIZERS.register_module()
class MultiBranchRecognizerGCN(BaseRecognizer):
    """Run independent ProtoGCN branches in one model and ensemble outputs.

    Input channels must be ordered exactly as ``branch_channels``. For the
    CARE-PD four-stream model this is ``[3, 3, 1, 3]`` for joint, joint
    motion, angle and bone respectively.
    """

    def __init__(self,
                 branches,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        nn.Module.__init__(self)
        self.branches = nn.ModuleList()
        self.branch_channels = []
        self.branch_names = []
        for index, branch_cfg in enumerate(branches):
            branch_cfg = branch_cfg.copy()
            name = branch_cfg.pop('name', f'branch_{index}')
            in_channels = branch_cfg.pop('in_channels')
            backbone_cfg = branch_cfg.pop('backbone')
            if branch_cfg:
                raise ValueError(
                    f'Unsupported options in branch {name}: {sorted(branch_cfg)}')
            backbone_cfg = backbone_cfg.copy()
            backbone_cfg['in_channels'] = in_channels
            self.branches.append(builder.build_backbone(backbone_cfg))
            self.branch_channels.append(in_channels)
            self.branch_names.append(name)

        self.cls_head = builder.build_head(cls_head) if cls_head else None
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.max_testing_views = self.test_cfg.get('max_testing_views', None)
        self.init_weights()

    @property
    def with_cls_head(self):
        return self.cls_head is not None

    def init_weights(self):
        for branch in self.branches:
            branch.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def _forward_branches(self, keypoint):
        if keypoint.dim() != 5:
            raise ValueError(
                f'Expected N,M,T,V,C input after clip flattening, got {tuple(keypoint.shape)}')
        required_channels = sum(self.branch_channels)
        if keypoint.shape[-1] != required_channels:
            raise ValueError(
                f'Expected {required_channels} input channels for '
                f'{self.branch_names}, got {keypoint.shape[-1]}')

        inputs = torch.split(keypoint, self.branch_channels, dim=-1)
        features = []
        graphs = []
        for name, branch, branch_input in zip(self.branch_names, self.branches, inputs):
            feature, graph = branch(branch_input)
            logger.debug('MultiBranch %s: input=%s output=%s', name,
                         tuple(branch_input.shape), tuple(feature.shape))
            features.append(feature)
            graphs.append(graph)
        return features, graphs

    def forward_train(self, keypoint, label, **kwargs):
        if keypoint.shape[1] != 1:
            raise ValueError(
                f'Training expects one clip, got {keypoint.shape[1]} clips')
        keypoint = keypoint[:, 0]
        features, graphs = self._forward_branches(keypoint)
        branch_scores = self.cls_head(features)
        gt_label = label.squeeze(-1)
        return self.cls_head.loss(branch_scores, graphs, gt_label)

    def forward_test(self, keypoint, **kwargs):
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])
        features, _ = self._forward_branches(keypoint)
        branch_scores = self.cls_head(features)
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        branch_probabilities = []
        for score in branch_scores:
            score = score.reshape(bs, nc, score.shape[-1])
            branch_probabilities.append(self.average_clip(score))
        ensemble_probability = self.cls_head.ensemble(branch_probabilities)
        return ensemble_probability.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)
        return self.forward_test(keypoint, **kwargs)
