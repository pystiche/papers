import itertools
from typing import Any, Iterator, List, Sized, Tuple

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import Sampler

__all__ = [
    "InfiniteCycleBatchSampler",
    "FiniteCycleBatchSampler",
    "DelayedExponentialLR",
]


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


class FiniteCycleBatchSampler(InfiniteCycleBatchSampler):
    def __init__(
        self, data_source: Sized, num_batches: int, batch_size: int = 1
    ) -> None:
        super().__init__(data_source, batch_size=batch_size)
        self.num_batches = num_batches

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        iterator = super().__iter__()
        for _ in range(self.num_batches):
            yield next(iterator)

    def __len__(self) -> int:
        return self.num_batches


class DelayedExponentialLR(ExponentialLR):
    r"""Decays the learning rate of each parameter group by gamma after the delay.

    Args:
        optimizer: Wrapped optimizer.
        gamma: Multiplicative factor of learning rate decay.
        delay: Number of epochs before the learning rate decays.
        **kwargs: Optional parameters for the
             :class:`~torch.optim.lr_scheduler.ExponentialLR`.
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, delay: int, **kwargs: Any
    ) -> None:
        self.last_epoch: int
        self.gamma: float
        self.base_lrs: List[float]

        self.delay = delay
        super().__init__(optimizer, gamma, **kwargs)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        exp = self.last_epoch - self.delay + 1
        if exp > 0:
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
        else:
            return self.base_lrs
