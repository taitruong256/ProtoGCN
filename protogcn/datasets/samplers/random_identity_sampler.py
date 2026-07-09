import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances,
                 num_replicas=1,
                 rank=0,
                 seed=0):
        if batch_size % num_instances != 0:
            raise ValueError('batch_size must be divisible by num_instances')

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed or 0
        self.epoch = 0

        self.index_dic = defaultdict(list)
        for index, info in enumerate(dataset.video_infos):
            self.index_dic[int(info['label'])].append(index)
        self.pids = sorted(self.index_dic.keys())
        self.length = self._estimate_length()

    def _estimate_length(self):
        total = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            total += max(num, self.num_instances)
        global_batch = self.batch_size * self.num_replicas
        total = int(math.ceil(total / global_batch) * global_batch)
        return total // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        per_rank_batches = []
        target_batches = max(1, self.length // self.batch_size)

        for replica_rank in range(self.num_replicas):
            local_rng = random.Random(rng.randint(0, 2**31 - 1) + replica_rank)
            batches = []
            while len(batches) < target_batches:
                selected_pids = local_rng.sample(
                    self.pids,
                    min(self.num_pids_per_batch, len(self.pids)))
                batch = []
                for pid in selected_pids:
                    indices = self.index_dic[pid]
                    if len(indices) >= self.num_instances:
                        batch.extend(local_rng.sample(indices, self.num_instances))
                    else:
                        batch.extend(local_rng.choices(indices, k=self.num_instances))

                if len(batch) < self.batch_size:
                    extra = local_rng.choices(batch, k=self.batch_size - len(batch))
                    batch.extend(extra)
                elif len(batch) > self.batch_size:
                    batch = batch[:self.batch_size]
                batches.extend(batch)
            per_rank_batches.append(batches[:self.length])

        return iter(per_rank_batches[self.rank])

    def __len__(self):
        return self.length
