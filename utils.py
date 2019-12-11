import random
from collections import defaultdict

import torch


class RunningOp:
    def __init__(self, n, op):
        self.x = []
        self.n = n
        self.op = op

    def __call__(self, x):
        self.x.append(x)
        self.x = self.x[-self.n:]
        return self.op(self.x)


def inf_shuffle(xs):
    while xs:
        random.shuffle(xs)
        for x in xs:
            yield x


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        indices = defaultdict(list)
        for i in range(0, len(dataset)):
            _, label = dataset[i]
            indices[label].append(i)
        self.indices = list(indices.values())

        self.n = max(len(ids) for ids in self.indices) * len(self.indices)

    def __iter__(self):
        m = 0
        for xs in zip(*(inf_shuffle(xs) for xs in self.indices)):
            for i in xs:  # yield one index of each label
                yield i
                m += 1
                if m >= self.n:
                    return

    def __len__(self):
        return self.n
