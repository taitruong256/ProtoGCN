import numpy as np

from .builder import DATASETS, build_dataset


@DATASETS.register_module()
class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be ``times`` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (dict): The config of the dataset to be repeated.
        times (int): Repeat times.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, dataset, times, test_mode=False):
        dataset['test_mode'] = test_mode
        self.dataset = build_dataset(dataset)
        self.times = times
        if hasattr(dataset, 'class_prob'):
            self.class_prob = dataset.class_prob

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get data."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length after repetition."""
        return self.times * self._ori_len


@DATASETS.register_module()
class ConcatDataset:
    """A wrapper of concatenated dataset.

    The length of concatenated dataset will be the sum of lengths of all
    datasets. This is useful when you want to train a model with multiple data
    sources.

    Args:
        datasets (list[dict]): The configs of the datasets.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self, datasets, test_mode=False):

        for item in datasets:
            item['test_mode'] = test_mode

        datasets = [build_dataset(cfg) for cfg in datasets]
        self.datasets = datasets
        self.lens = [len(x) for x in self.datasets]
        self.cumsum = np.cumsum(self.lens)
        self.video_infos = []
        for dataset in self.datasets:
            if hasattr(dataset, 'video_infos'):
                self.video_infos.extend(dataset.video_infos)
        self.num_classes = next(
            (getattr(dataset, 'num_classes', None) for dataset in self.datasets
             if getattr(dataset, 'num_classes', None) is not None),
            None,
        )

    def __getitem__(self, idx):
        """Get data."""
        dataset_idx = np.searchsorted(self.cumsum, idx, side='right')
        item_idx = idx if dataset_idx == 0 else idx - self.cumsum[dataset_idx]
        return self.datasets[dataset_idx][item_idx]

    def __len__(self):
        """Length after repetition."""
        return sum(self.lens)

    def evaluate(self, results, metrics=None, logger=None, return_confusion_matrix=False, **kwargs):
        from .remap_dataset import _evaluate_predictions

        return _evaluate_predictions(
            self.video_infos,
            results,
            self.num_classes,
            metrics=metrics,
            logger=logger,
            return_confusion_matrix=return_confusion_matrix,
            **kwargs,
        )
