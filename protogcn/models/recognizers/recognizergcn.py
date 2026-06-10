import numpy as np
import logging
import torch
import torch.nn as nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer

logger = logging.getLogger(__name__)


def _pool_features_before_head(x):
    """Pool backbone features to the same shape expected by the classifier."""
    if isinstance(x, tuple) or isinstance(x, list):
        x = torch.cat(x, dim=2)

    if len(x.shape) == 2:
        return x

    if len(x.shape) != 5:
        raise ValueError(f'Unsupported feature shape for extraction: {tuple(x.shape)}')

    pool = torch.nn.AdaptiveAvgPool2d(1)
    n, m, c, t, v = x.shape
    x = x.reshape(n * m, c, t, v)
    x = pool(x)
    x = x.reshape(n, m, c)
    x = x.mean(dim=1)
    return x


def _unwrap_meta(meta):
    if meta is None:
        return None
    if hasattr(meta, 'data'):
        return _unwrap_meta(meta.data)
    if isinstance(meta, (list, tuple)):
        return [_unwrap_meta(item) for item in meta]
    return meta


def _view_to_index(view, num_views):
    if isinstance(view, torch.Tensor):
        view = view.detach().cpu().view(-1)[0].item()
    elif isinstance(view, np.ndarray):
        view = np.asarray(view).reshape(-1)[0].item()

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


@RECOGNIZERS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 view_num=11,
                 view_loss_weight=1.0,
                 view_label_key='view',
                 **kwargs):
        self.view_num = view_num
        self.view_loss_weight = view_loss_weight
        self.view_label_key = view_label_key
        super().__init__(backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.view_loss = nn.CrossEntropyLoss()

    def _extract_view_labels(self, img_metas, device):
        img_metas = _unwrap_meta(img_metas)
        if img_metas is None:
            return None
        if isinstance(img_metas, dict):
            img_metas = [img_metas]
        elif not isinstance(img_metas, (list, tuple)):
            img_metas = [img_metas]

        view_labels = []
        for meta in img_metas:
            if isinstance(meta, dict):
                view = meta.get(self.view_label_key, meta.get('view'))
            else:
                view = meta
            view_labels.append(_view_to_index(view, self.view_num))
        return torch.tensor(view_labels, device=device, dtype=torch.long)

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        logger.debug("RecognizerGCN.forward_train: keypoint=%s label=%s", tuple(keypoint.shape), tuple(label.shape))
        keypoint = keypoint[:, 0]

        losses = dict()
        x, get_graph = self.extract_feat(keypoint)
        logger.debug(
            "RecognizerGCN.forward_train: backbone_out=%s graph=%s",
            tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__,
            tuple(get_graph.shape) if isinstance(get_graph, torch.Tensor) else type(get_graph).__name__,
        )
        cls_score = self.cls_head(x)
        logger.debug(
            "RecognizerGCN.forward_train: cls_score=%s",
            tuple(cls_score.shape) if isinstance(cls_score, torch.Tensor) else type(cls_score).__name__,
        )
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, get_graph, gt_label)
        losses.update(loss)

        view_logits = getattr(self.backbone, 'view_logits', None)
        if view_logits is None:
            raise RuntimeError('Backbone did not produce view logits. Check unit_gcn/view_num configuration.')

        view_label = self._extract_view_labels(kwargs.get('img_metas'), device=gt_label.device)
        if view_label is None:
            raise ValueError(
                'img_metas is required to train the view classifier. '
                'Make sure the pipeline Collect step keeps the `view` meta key.'
            )

        if view_logits.size(0) != view_label.size(0):
            raise ValueError(
                f'View logits batch size {view_logits.size(0)} does not match '
                f'view labels batch size {view_label.size(0)}.'
            )

        losses['loss_view'] = self.view_loss(view_logits, view_label) * self.view_loss_weight
        with torch.no_grad():
            losses['view_acc'] = (view_logits.argmax(dim=1) == view_label).float().mean()

        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        logger.debug("RecognizerGCN.forward_test: keypoint=%s", tuple(keypoint.shape))
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        x, get_graph = self.extract_feat(keypoint)
        logger.debug(
            "RecognizerGCN.forward_test: backbone_out=%s graph=%s",
            tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__,
            tuple(get_graph.shape) if isinstance(get_graph, torch.Tensor) else type(get_graph).__name__,
        )
        feat_ext = self.test_cfg.get('feat_ext', False)
        return_view_score = self.test_cfg.get('return_view_score', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if feat_ext:
                feat = _pool_features_before_head(x)
                feat = feat.reshape(bs, nc, -1).mean(dim=1)
                if not return_view_score:
                    return feat.data.cpu().numpy().astype(np.float32)

                view_logits = getattr(self.backbone, 'view_logits', None)
                if view_logits is None:
                    raise RuntimeError('Backbone did not produce view logits.')
                view_score = view_logits.reshape(bs, nc, -1).mean(dim=1)
                feat = feat.data.cpu().numpy().astype(np.float32)
                view_score = view_score.data.cpu().numpy().astype(np.float32)
                return feat, view_score

            if pool_opt == 'all':
                pool_opt = 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x.reshape(bs, nc, *x.shape[1:]).mean(dim=1)
                return x.data.cpu().numpy().astype(np.float16)

        cls_score = self.cls_head(x)
        logger.debug(
            "RecognizerGCN.forward_test: cls_score=%s",
            tuple(cls_score.shape) if isinstance(cls_score, torch.Tensor) else type(cls_score).__name__,
        )
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):

        return self.backbone(keypoint)
