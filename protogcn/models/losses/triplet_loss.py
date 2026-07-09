import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


def _gather_if_distributed(tensor, requires_grad=True):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    rank = dist.get_rank()
    gathered = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    if requires_grad:
        gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


@LOSSES.register_module()
class TripletLoss(BaseWeightedLoss):

    def __init__(self, margin=0.3, is_hard_loss=True, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight)
        self.margin = margin
        self.is_hard_loss = is_hard_loss

    def _compute_distance(self, x, y):
        x2 = torch.sum(x ** 2, dim=-1).unsqueeze(2)
        y2 = torch.sum(y ** 2, dim=-1).unsqueeze(1)
        inner = x.matmul(y.transpose(1, 2))
        dist_map = torch.sqrt(F.relu(x2 + y2 - 2 * inner))
        return dist_map

    def _convert_to_triplets(self, row_labels, col_labels, dist_map):
        matches = row_labels.unsqueeze(1).eq(col_labels.unsqueeze(0))
        diffs = ~matches
        p, n, _ = dist_map.size()
        ap_dist = dist_map[:, matches].view(p, n, -1, 1)
        an_dist = dist_map[:, diffs].view(p, n, 1, -1)
        return ap_dist, an_dist

    def _avg_non_zero_reducer(self, loss):
        eps = 1e-9
        loss_sum = loss.sum(dim=-1)
        loss_num = (loss != 0).sum(dim=-1).float()
        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def _batch_hard_triplet_loss(self, ap_dist, an_dist, dist_map):
        ap_dist = ap_dist.max(dim=-2)[0].view(dist_map.size(0), dist_map.size(1))
        an_dist = an_dist.min(dim=-1)[0].view(dist_map.size(0), dist_map.size(1))
        loss = F.relu(ap_dist - an_dist + self.margin)
        hard_loss, _ = self._avg_non_zero_reducer(loss)
        return hard_loss

    def _batch_all_triplet_loss(self, ap_dist, an_dist, dist_map):
        dist_diff = (ap_dist - an_dist).view(dist_map.size(0), -1)
        loss = F.relu(dist_diff + self.margin)
        loss_avg, _ = self._avg_non_zero_reducer(loss)
        return loss_avg

    def _forward(self, embeddings, labels):
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(-1)
        if embeddings.ndim != 3:
            raise ValueError(
                f'TripletLoss expects embeddings of shape [n, c] or [n, c, p], got {tuple(embeddings.shape)}')

        if labels.ndim > 1:
            labels = labels.view(-1)
        labels = labels.long()

        embeddings = _gather_if_distributed(embeddings, requires_grad=True)
        labels = _gather_if_distributed(labels, requires_grad=False)

        embeddings = embeddings.permute(2, 0, 1).contiguous().float()
        dist_map = self._compute_distance(embeddings, embeddings)
        ap_dist, an_dist = self._convert_to_triplets(labels, labels, dist_map)

        if self.is_hard_loss:
            loss = self._batch_hard_triplet_loss(ap_dist, an_dist, dist_map)
        else:
            loss = self._batch_all_triplet_loss(ap_dist, an_dist, dist_map)
        return loss.mean()
