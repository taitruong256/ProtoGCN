import logging
import os
import os.path as osp
import shutil
import threading

import numpy as np
import torch.distributed as dist
from mmcv.engine import multi_gpu_test
from mmcv.runner import DistEvalHook as BasicDistEvalHook
from mmcv.runner import get_dist_info


class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', seg_interval=None, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        self.seg_interval = seg_interval
        self._log_next_train_iter = False
        self._configured_save_best = save_best
        if seg_interval is not None:
            assert isinstance(seg_interval, list)
            for i, tup in enumerate(seg_interval):
                assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
                if i < len(seg_interval) - 1:
                    assert tup[1] == seg_interval[i + 1][0]
            assert self.by_epoch
        assert self.start is None

    def _find_n(self, runner):
        current = runner.epoch
        for seg in self.seg_interval:
            if current >= seg[0] and current < seg[1]:
                return seg[2]
        return None

    def _should_evaluate(self, runner):
        if self.seg_interval is None:
            return super()._should_evaluate(runner)
        n = self._find_n(runner)
        assert n is not None
        return self.every_n_epochs(runner, n)

    def before_train_iter(self, runner):
        if not self._log_next_train_iter:
            return
        rank, _ = get_dist_info()
        if rank == 0:
            runner.logger.info('Training resumed after validation at iter=%s.',
                               runner.iter + 1)
        self._log_next_train_iter = False

    def after_train_iter(self, runner):
        if not self._should_evaluate(runner):
            return
        self._do_evaluate(runner)

    def _update_best_checkpoint(self, runner, eval_results):
        key_indicator = getattr(self, 'key_indicator', None)
        if key_indicator in (None, 'auto'):
            key_indicator = self._configured_save_best
        if key_indicator in (None, 'auto'):
            key_indicator = next(iter(eval_results))

        if key_indicator not in eval_results:
            runner.logger.warning(
                'Skip best checkpoint because metric %s is missing.',
                key_indicator)
            return

        rule = getattr(self, 'rule', None)
        if rule is None or rule == 'auto':
            rule = 'less' if any(key in key_indicator for key in self.less_keys) else 'greater'

        score = eval_results[key_indicator]
        best_score = getattr(self, 'best_score', None)
        if best_score is None:
            best_score = np.inf if rule == 'less' else -np.inf

        is_better = score < best_score if rule == 'less' else score > best_score
        if not is_better:
            return

        current = runner.iter + 1
        ckpt_path = osp.join(runner.work_dir, f'iter_{current}.pth')
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(runner.work_dir, 'latest.pth')
        if not osp.exists(ckpt_path):
            runner.logger.warning(
                'Skip best checkpoint because no checkpoint exists for iter %s.',
                current)
            return

        previous_best = getattr(self, 'best_ckpt_path', None)
        if previous_best and osp.exists(previous_best):
            os.remove(previous_best)

        best_name = f'best_{key_indicator}_iter_{current}.pth'
        best_path = osp.join(runner.work_dir, best_name)
        shutil.copyfile(ckpt_path, best_path)
        self.best_ckpt_path = best_path
        self.best_score = score
        runner.meta.setdefault('hook_msgs', {})
        runner.meta['hook_msgs']['best_score'] = score
        runner.meta['hook_msgs']['best_ckpt'] = best_path
        runner.logger.info('Now best checkpoint is saved as %s.', best_name)
        runner.logger.info('Best %s is %.4f at %s iter.',
                           key_indicator, score, current)

    def _do_evaluate(self, runner):
        rank, _ = get_dist_info()
        heartbeat_stop = threading.Event()
        heartbeat_thread = None

        if rank == 0:
            logger = getattr(runner, 'logger', logging.getLogger(__name__))

            def _heartbeat():
                while not heartbeat_stop.wait(60):
                    logger.info('Validation is still running ...')

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()

        try:
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=getattr(self, 'tmpdir', None),
                gpu_collect=getattr(self, 'gpu_collect', False))

            if rank == 0:
                eval_kwargs = getattr(self, 'eval_kwargs', {})
                eval_results = self.dataloader.dataset.evaluate(
                    results, logger=runner.logger, **eval_kwargs)
                for name, val in eval_results.items():
                    runner.log_buffer.output[name] = val
                runner.log_buffer.ready = True
                if self._configured_save_best:
                    self._update_best_checkpoint(runner, eval_results)
                logger.info('Validation hook finished metrics; synchronizing ranks ...')
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            runner.model.train()
            self._log_next_train_iter = True
            if rank == 0:
                logger.info('Validation hook finished; training mode restored.')
            return None
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])

    return mean_class_acc


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def mean_average_precision(scores, labels):
    """Mean average precision for multi-label recognition.

    Args:
        scores (list[np.ndarray]): Prediction scores of different classes for
            each sample.
        labels (list[np.ndarray]): Ground truth many-hot vector for each
            sample.

    Returns:
        np.float: The mean average precision.
    """
    results = []
    scores = np.stack(scores).T
    labels = np.stack(labels).T

    for score, label in zip(scores, labels):
        precision, recall, _ = binary_precision_recall_curve(score, label)
        ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        results.append(ap)
    results = [x for x in results if not np.isnan(x)]
    if results == []:
        return np.nan
    return np.mean(results)


def binary_precision_recall_curve(y_score, y_true):
    """Calculate the binary precision recall curve at step thresholds.

    Args:
        y_score (np.ndarray): Prediction scores for each class.
            Shape should be (num_classes, ).
        y_true (np.ndarray): Ground truth many-hot vector.
            Shape should be (num_classes, ).

    Returns:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.
        thresholds (np.ndarray): Different thresholds at which precision and
            recall are tested.
    """
    assert isinstance(y_score, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert y_score.shape == y_true.shape

    # make y_true a boolean vector
    y_true = (y_true == 1)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind='mergesort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # There may be ties in values, therefore find the `distinct_value_inds`
    distinct_value_inds = np.where(np.diff(y_score))[0]
    threshold_inds = np.r_[distinct_value_inds, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_inds]
    fps = 1 + threshold_inds - tps
    thresholds = y_score[threshold_inds]

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
