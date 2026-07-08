import math
import torch
from collections import defaultdict
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from 'torch.utils.data.DistributedSampler'.

    Samples are sampled with a class specific probability (class_prob). This sampler is only applicable to single class
    recognition dataset. This sampler is also compatible with RepeatDataset.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 class_prob=None,
                 shuffle=True,
                 seed=0):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        if class_prob is not None:
            if isinstance(class_prob, list):
                class_prob = {i: n for i, n in enumerate(class_prob)}
            assert isinstance(class_prob, dict)
        self.class_prob = class_prob
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_prob = self.class_prob
        dataset_name = type(self.dataset).__name__
        dataset = self.dataset if dataset_name != 'RepeatDataset' else self.dataset.dataset
        times = 1
        if dataset_name == 'RepeatDataset':
            times = self.dataset.times
            class_prob = {k: v * times for k, v in class_prob.items()}

        labels = [x['label'] for x in dataset.video_infos]
        samples = defaultdict(list)
        for i, lb in enumerate(labels):
            samples[lb].append(i)

        indices = []
        for class_idx, class_indices in samples.items():
            mul = class_prob.get(class_idx, times)
            for i in range(int(mul // 1)):
                indices.extend(class_indices)
            rem = int((mul % 1) * len(class_indices))
            inds = torch.randperm(len(class_indices), generator=g).tolist()
            indices.extend([class_indices[inds[i]] for i in range(rem)])

        if self.shuffle:
            shuffle = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle]

        # reset num_samples and total_size here.
        self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class TripletBatchSampler(Sampler):
    """Infinite P x K sampler for metric-learning batches.

    Each yielded batch contains P labels and K samples for each label. In
    distributed training every rank builds the same global batch, then receives
    its rank slice.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=1,
                 rank=0,
                 batch_shuffle=False,
                 seed=0):
        if not isinstance(batch_size, (list, tuple)) or len(batch_size) != 2:
            raise ValueError(f'batch_size should be [P, K], got {batch_size}')

        self.dataset = dataset
        self.num_labels = int(batch_size[0])
        self.num_instances = int(batch_size[1])
        self.total_batch_size = self.num_labels * self.num_instances
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_shuffle = batch_shuffle
        self.seed = seed if seed is not None else 0
        self.epoch = 0

        if self.num_labels <= 1 or self.num_instances <= 1:
            raise ValueError('TripletBatchSampler requires P > 1 and K > 1.')
        if self.total_batch_size % self.num_replicas != 0:
            raise ValueError(
                f'World size ({self.num_replicas}) must divide '
                f'P x K ({self.num_labels} x {self.num_instances}).')

        labels = [item['label'] for item in self._get_video_infos(dataset)]
        self.indices_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            self.indices_dict[int(label)].append(idx)
        self.label_set = sorted(self.indices_dict)
        if len(self.label_set) < self.num_labels:
            raise ValueError(
                f'Dataset has {len(self.label_set)} labels, fewer than '
                f'P={self.num_labels}.')

    @staticmethod
    def _get_video_infos(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if not hasattr(dataset, 'video_infos'):
            raise AttributeError(
                'TripletBatchSampler expects dataset.video_infos with labels.')
        return dataset.video_infos

    def _sample(self, values, k, generator):
        values = list(values)
        if len(values) < k:
            pos = torch.randint(len(values), (k,), generator=generator).tolist()
        else:
            pos = torch.randperm(len(values), generator=generator)[:k].tolist()
        return [values[i] for i in pos]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        while True:
            sample_indices = []
            pid_list = self._sample(self.label_set, self.num_labels, generator)
            for pid in pid_list:
                sample_indices.extend(
                    self._sample(self.indices_dict[pid], self.num_instances,
                                 generator))

            if self.batch_shuffle:
                order = torch.randperm(
                    len(sample_indices), generator=generator).tolist()
                sample_indices = [sample_indices[i] for i in order]

            yield sample_indices[self.rank::self.num_replicas]

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = epoch
