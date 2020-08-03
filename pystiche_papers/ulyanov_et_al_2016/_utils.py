from typing import Any, List

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from pystiche import enc
from pystiche.image import transforms

__all__ = [
    "multi_layer_encoder",
    "preprocessor",
    "postprocessor",
    "optimizer",
    "DelayedExponentialLR",
    "lr_scheduler",
]


def multi_layer_encoder() -> enc.VGGMultiLayerEncoder:
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def optimizer(
    transformer: nn.Module, impl_params: bool = True, instance_norm: bool = True
) -> optim.Adam:
    lr = 1e-3 if impl_params and instance_norm else 1e-1
    return optim.Adam(transformer.parameters(), lr=lr)


class DelayedExponentialLR(ExponentialLR):
    last_epoch: int
    gamma: float
    base_lrs: List[float]

    def __init__(
        self, optimizer: Optimizer, gamma: float, delay: int, **kwargs: Any
    ) -> None:
        self.delay = delay
        super().__init__(optimizer, gamma, **kwargs)

    def get_lr(self) -> List[float]:  # type: ignore[override]
        exp = self.last_epoch - self.delay + 1
        if exp > 0:
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def lr_scheduler(optimizer: Optimizer, impl_params: bool = True,) -> ExponentialLR:
    return (
        ExponentialLR(optimizer, 0.8)
        if impl_params
        else DelayedExponentialLR(optimizer, 0.7, 5)
    )
