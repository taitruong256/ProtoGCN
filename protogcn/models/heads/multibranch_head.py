import logging

import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from .base import BaseHead
from ..builder import HEADS


logger = logging.getLogger(__name__)


@HEADS.register_module()
class MultiBranchHead(BaseHead):
    """Fuse embeddings from multiple skeleton branches before classification.

    Each branch returns ``N, M, C, T, V`` features. The features are pooled
    independently, concatenated, projected by ``fusion_fc`` and classified.
    The first branch graph is used by the existing class-specific contrastive
    loss, preserving the loss used by ``SimpleHead``.
    """

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 branch_channels=256,
                 fusion_channels=256,
                 weight=0.2,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 dropout=0.0,
                 init_std=0.01,
                 **kwargs):
        super().__init__(joint_cfg, num_classes, fusion_channels, weight,
                         loss_cls, **kwargs)
        if isinstance(branch_channels, int):
            branch_channels = [branch_channels]
        self.branch_channels = list(branch_channels)
        self.fusion_channels = fusion_channels
        self.init_std = init_std
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.fusion_fc = nn.Linear(sum(self.branch_channels), fusion_channels)
        self.fc_cls = nn.Linear(fusion_channels, num_classes)

    @staticmethod
    def _pool_branch(x):
        if x.dim() != 5:
            raise ValueError(
                f'MultiBranchHead expects N,M,C,T,V branch features, got {tuple(x.shape)}')
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v).mean(dim=(2, 3))
        return x.reshape(n, m, c).mean(dim=1)

    def init_weights(self):
        normal_init(self.fusion_fc, std=self.init_std)
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, branch_features):
        if not isinstance(branch_features, (list, tuple)):
            raise TypeError('MultiBranchHead expects a list of branch features')
        if len(branch_features) != len(self.branch_channels):
            raise ValueError(
                f'Expected {len(self.branch_channels)} branches, got {len(branch_features)}')

        pooled = [self._pool_branch(feature) for feature in branch_features]
        for feature, expected in zip(pooled, self.branch_channels):
            if feature.shape[1] != expected:
                raise ValueError(
                    f'Branch embedding has {feature.shape[1]} channels, expected {expected}')
        fused = self.fusion_fc(torch.cat(pooled, dim=1))
        fused = torch.relu(fused)
        if self.dropout is not None:
            fused = self.dropout(fused)
        return self.fc_cls(fused)

