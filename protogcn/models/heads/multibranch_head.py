import logging

import torch
import torch.nn as nn

from .simple_head import SimpleHead
from ..builder import HEADS


logger = logging.getLogger(__name__)


@HEADS.register_module()
class MultiBranchHead(nn.Module):
    """Independent classifiers for multiple skeleton feature branches.

    There is deliberately no learnable fusion layer.  Every branch owns its
    classifier and loss; their probabilities are averaged only for
    validation/test prediction.
    """

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 branch_channels=256,
                 branch_names=None,
                 ensemble_weights=None,
                 weight=0.2,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 dropout=0.0,
                 init_std=0.01,
                 **kwargs):
        super().__init__()
        if isinstance(branch_channels, int):
            branch_channels = [branch_channels]
        self.branch_channels = list(branch_channels)
        self.branch_names = list(branch_names or [
            f'branch_{index}' for index in range(len(self.branch_channels))
        ])
        if len(self.branch_names) != len(self.branch_channels):
            raise ValueError('branch_names and branch_channels must have equal length')

        if ensemble_weights is None:
            ensemble_weights = [1.0] * len(self.branch_channels)
        if len(ensemble_weights) != len(self.branch_channels):
            raise ValueError(
                'ensemble_weights and branch_channels must have equal length')
        ensemble_weights = torch.tensor(ensemble_weights, dtype=torch.float32)
        if torch.any(ensemble_weights < 0) or ensemble_weights.sum() <= 0:
            raise ValueError('ensemble_weights must be non-negative with a positive sum')
        self.register_buffer(
            'ensemble_weights', ensemble_weights / ensemble_weights.sum())

        self.heads = nn.ModuleList([
            SimpleHead(
                joint_cfg=joint_cfg,
                num_classes=num_classes,
                in_channels=in_channels,
                weight=weight,
                loss_cls=loss_cls.copy(),
                dropout=dropout,
                init_std=init_std,
                **kwargs)
            for in_channels in self.branch_channels
        ])
        self.num_classes = num_classes

    def init_weights(self):
        for head in self.heads:
            head.init_weights()

    def forward(self, branch_features):
        if not isinstance(branch_features, (list, tuple)):
            raise TypeError('MultiBranchHead expects a list of branch features')
        if len(branch_features) != len(self.heads):
            raise ValueError(
                f'Expected {len(self.heads)} branches, got {len(branch_features)}')
        return [head(feature) for head, feature in zip(self.heads, branch_features)]

    def loss(self, branch_scores, branch_graphs, label, **kwargs):
        if len(branch_scores) != len(self.heads):
            raise ValueError('The number of branch scores does not match the heads')
        if len(branch_graphs) != len(self.heads):
            raise ValueError('The number of branch graphs does not match the heads')

        losses = {}
        num_branches = len(self.heads)
        for name, head, score, graph in zip(
                self.branch_names, self.heads, branch_scores, branch_graphs):
            branch_losses = head.loss(score, graph, label, **kwargs)
            for key, value in branch_losses.items():
                # BaseRecognizer sums keys containing "loss". Dividing each
                # branch loss keeps the total at the mean of four losses.
                if 'loss' in key:
                    value = value / num_branches
                losses[f'{name}_{key}'] = value
        return losses

    def ensemble(self, branch_probabilities):
        """Average already-normalized class probabilities across branches."""
        if len(branch_probabilities) != len(self.heads):
            raise ValueError(
                f'Expected {len(self.heads)} branch predictions, '
                f'got {len(branch_probabilities)}')
        stacked = torch.stack(branch_probabilities, dim=0)
        weights = self.ensemble_weights.to(
            device=stacked.device, dtype=stacked.dtype)
        return (stacked * weights[:, None, None]).sum(dim=0)
