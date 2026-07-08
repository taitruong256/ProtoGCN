import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class TripletLoss(BaseWeightedLoss):
    """Batch-hard / batch-all triplet loss used by GaitTR.

    Args:
        margin (float): Triplet margin.
        is_hard_loss (bool): If true, use hardest positive and hardest
            negative per anchor. Otherwise use all valid triplets.
        loss_weight (float): Scalar loss weight.
    """

    def __init__(self, margin=0.3, is_hard_loss=True, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.margin = margin
        self.is_hard_loss = is_hard_loss

    def _forward(self, embeddings, labels):
        # GaitTR format: embeddings [N, C, P] -> [P, N, C].
        embeddings = embeddings.permute(2, 0, 1).contiguous().float()
        labels = labels.view(-1)

        dist = self.compute_distance(embeddings, embeddings)
        ap_dist, an_dist = self.convert_to_triplets(labels, labels, dist)

        if self.is_hard_loss:
            loss = self.batch_hard_triplet_loss(ap_dist, an_dist, dist)
        else:
            loss = self.batch_all_triplet_loss(ap_dist, an_dist, dist)
        return loss.mean()

    def batch_hard_triplet_loss(self, ap_dist, an_dist, dist):
        ap_dist = ap_dist.max(-2)[0].view(dist.size(0), dist.size(1))
        an_dist = an_dist.min(-1)[0].view(dist.size(0), dist.size(1))
        loss = F.relu(ap_dist - an_dist + self.margin)
        loss_avg, _ = self.avg_non_zero_reducer(loss)
        return loss_avg

    def batch_all_triplet_loss(self, ap_dist, an_dist, dist):
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)
        loss_avg, _ = self.avg_non_zero_reducer(loss)
        return loss_avg

    @staticmethod
    def avg_non_zero_reducer(loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()
        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    @staticmethod
    def compute_distance(x, y):
        x2 = torch.sum(x**2, -1).unsqueeze(2)
        y2 = torch.sum(y**2, -1).unsqueeze(1)
        inner = x.matmul(y.transpose(1, 2))
        dist = x2 + y2 - 2 * inner
        return torch.sqrt(F.relu(dist))

    @staticmethod
    def convert_to_triplets(row_labels, col_labels, dist):
        matches = (row_labels.unsqueeze(1) == col_labels.unsqueeze(0)).bool()
        differs = torch.logical_not(matches)
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, differs].view(p, n, 1, -1)
        return ap_dist, an_dist
