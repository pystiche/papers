from typing import Iterator, Union

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from pystiche.optim.meter import FloatMeter

from ..data.utils import DelayedExponentialLR


def optimizer(
    params: Union[nn.Module, Iterator[torch.nn.parameter.Parameter]]
) -> Optimizer:
    r"""Optimizer from :cite:`SKL+2018`.

    Args:
        params: Parameter to be optimized.

    """
    if isinstance(params, nn.Module):
        params = params.parameters()
    return optim.Adam(params, lr=2e-4)


def lr_scheduler(optimizer: Optimizer) -> ExponentialLR:
    r"""Learning rate scheduler from :cite:`SKL+2018`.

    Args:
        optimizer: Wrapped optimizer.

    """
    return DelayedExponentialLR(optimizer, gamma=0.1, delay=2)


class ExponentialMovingAverageMeter(FloatMeter):
    def __init__(
        self,
        name: str,
        init_val: float,
        smoothing_factor: float = 0.05,
        fmt: str = "{:f}",
    ) -> None:
        super().__init__(name)
        self.smoothing_factor = smoothing_factor
        self.last_val = init_val
        self.fmt = fmt

    def calculate_val(self, val: Union[torch.Tensor, float]) -> float:
        if isinstance(val, torch.Tensor):
            val = val.item()
        return (
            self.last_val * (1.0 - self.smoothing_factor) + self.smoothing_factor * val
        )

    def update(self, new_val: Union[torch.Tensor, float]) -> None:  # type: ignore[override]
        super().update(self.calculate_val(new_val))

    def global_avg(self) -> float:
        return self.last_val

    def __str__(self) -> str:
        def format(val: float) -> str:
            return self.fmt.format(val)

        val = format(self.last_val)
        info = f"{val}"
        return f"{self.name} {info}"
