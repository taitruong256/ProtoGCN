import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS, build_loss
from ..losses.Class_Specific_Contrastive_Loss import Class_Specific_Contrastive_Loss

logger = logging.getLogger(__name__)


def _infer_graph_channels(joint_cfg):
    if joint_cfg == 'nturgb+d':
        return 625
    if joint_cfg == 'coco':
        return 289
    if joint_cfg == 'coco_new':
        return 400
    raise ValueError(f'Unsupported joint_cfg for CSC loss: {joint_cfg}')


@HEADS.register_module()
class GaitHead(nn.Module):

    def __init__(self,
                 joint_cfg,
                 num_classes,
                 in_channels,
                 weight,
                 triplet_loss=dict(type='TripletLoss', margin=0.2, loss_weight=1.0),
                 dropout=0.,
                 init_std=0.01,
                 embedding_dim=None,
                 normalize=True,
                 mode='GCN',
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.weight = weight
        self.init_std = init_std
        self.normalize = normalize
        self.mode = mode
        self.embedding_dim = embedding_dim or in_channels

        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.proj = None
        if self.embedding_dim != in_channels:
            self.proj = nn.Linear(in_channels, self.embedding_dim)

        self.triplet_loss = build_loss(triplet_loss)
        self.csc_loss = Class_Specific_Contrastive_Loss(
            num_classes, _infer_graph_channels(joint_cfg))

    def init_weights(self):
        if self.proj is not None:
            normal_init(self.proj, std=self.init_std)

    def _pool(self, x):
        if isinstance(x, list):
            x = torch.stack([item.mean(dim=0) for item in x], dim=0)

        if len(x.shape) == 2:
            return x

        if len(x.shape) != 5:
            raise ValueError(f'Unsupported input shape for GaitHead: {tuple(x.shape)}')

        if self.mode != 'GCN':
            raise ValueError(f'Unsupported mode for GaitHead: {self.mode}')

        pool = nn.AdaptiveAvgPool2d(1)
        n, m, c, t, v = x.shape
        x = x.reshape(n * m, c, t, v)
        x = pool(x)
        x = x.reshape(n, m, c).mean(dim=1)
        return x

    def forward(self, x):
        logger.debug("GaitHead.forward: in=%s", tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__)
        x = self._pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.proj is not None:
            x = self.proj(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        logger.debug("GaitHead.forward: out=%s", tuple(x.shape))
        return x

    def loss(self, embeddings, reconstructed_graph, label, **kwargs):
        losses = dict()
        if label.ndim > 1:
            label = label.view(-1)

        losses['loss_triplet'] = self.triplet_loss(embeddings, label)
        losses['loss_csc'] = self.weight * self.csc_loss(reconstructed_graph, label)
        return losses
