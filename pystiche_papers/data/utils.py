import itertools
from abc import ABC, abstractmethod
from typing import cast, Iterator, List, Sized, Tuple

import more_itertools

import torch
from torch.utils.data import Sampler

__all__ = [
    "NumIterationsBatchSampler",
    "SequentialNumIterationsBatchSampler",
    "RandomNumIterationsBatchSampler",
    "FiniteCycleBatchSampler",
]


class NumIterationsBatchSampler(Sampler, ABC):
    def __init__(
        self,
        data_source: Sized,
        num_iterations: int,
        batch_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        return (
            self.num_iterations // self.batch_size
            if self.drop_last
            else (self.num_iterations + self.batch_size - 1) // self.batch_size
        )

    def _next_batch(self, iterator: Iterator[int]) -> List[int]:
        return more_itertools.take(self.batch_size, iterator)

    def __iter__(self) -> Iterator[List[int]]:
        iterator = self.infinite_iterator()
        for _ in range(len(self)):
            yield self._next_batch(iterator)

    @abstractmethod
    def infinite_iterator(self) -> Iterator[int]:
        pass


class SequentialNumIterationsBatchSampler(NumIterationsBatchSampler):
    def infinite_iterator(self) -> Iterator[int]:
        return itertools.cycle(range(len(self.data_source)))


class RandomNumIterationsBatchSampler(NumIterationsBatchSampler):
    def infinite_iterator(self) -> Iterator[int]:
        high = len(self.data_source)
        while True:
            yield cast(int, torch.randint(high=high, size=(), dtype=torch.int64).item())


# TODO: remove this as is serves no purpose anymore
class InfiniteCycleBatchSampler(Sampler):
    def __init__(self, data_source: Sized, batch_size: int = 1) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        def nextn(iterator: Iterator[int], n: int) -> Iterator[int]:
            for _ in range(n):
                yield next(iterator)

        iterator = itertools.cycle(range(len(self.data_source)))
        while True:
            yield tuple(nextn(iterator, self.batch_size))


# TODO: remove this as is serves no purpose anymore
class FiniteCycleBatchSampler(SequentialNumIterationsBatchSampler):
    def __init__(self, data_source: Sized, num_batches: int, batch_size: int = 1):
        self.num_batches = num_batches
        num_iterations = num_batches * batch_size
        super().__init__(
            data_source, num_iterations, batch_size=batch_size, drop_last=False
        )

    def __iter__(self) -> Iterator[Tuple[int, ...]]:  # type: ignore[override]
        for batch in super().__iter__():
            yield tuple(batch)
