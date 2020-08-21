from typing import Any, List, cast

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
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(  # type: ignore[attr-defined]
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )
    return cast(enc.VGGMultiLayerEncoder, multi_layer_encoder)


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def optimizer(
    transformer: nn.Module, impl_params: bool = True, instance_norm: bool = True
) -> optim.Adam:
    # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L29
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
    # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L260
    # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L201
    return (
        ExponentialLR(optimizer, 0.8)
        if impl_params
        else DelayedExponentialLR(optimizer, 0.7, 5)
    )
