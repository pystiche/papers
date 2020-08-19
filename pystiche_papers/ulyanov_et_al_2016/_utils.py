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
    r"""Multi-layer encoder from :cite:`ULVL2016`."""
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
    r"""Optimizer from :cite:`ULVL2016`.

    Args:
        transformer: Transformer to be optimized.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            :ref:`here <table-branches-ulyanov_et_al_2016>`.

    """
    # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L29
    lr = 1e-3 if impl_params and instance_norm else 1e-1
    return optim.Adam(transformer.parameters(), lr=lr)


class DelayedExponentialLR(ExponentialLR):
    r"""Decays the learning rate of each parameter group by gamma after the delay.

    Args:
        optimizer: Wrapped optimizer.
        gamma: Multiplicative factor of learning rate decay.
        delay: Number of epochs before the learning rate is reduced with each epoch.
        **kwargs: Optional parameters for the
             :class:`~torch.optim.lr_scheduler.ExponentialLR`.
    """
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
    r"""Learning rate scheduler from :cite:`ULVL2016`.

    Args:
        optimizer: Wrapped optimizer.
        impl_params: If ``True``, an :class:`~torch.optim.lr_scheduler.ExponentialLR`
        with ``gamma==0.8`` is used instead of a
        :func:`~pystiche_papers.ulyanov_et_al_2016.DelayedExponentialLR` with
        ``gamma==0.7`` and ``delay==5``.
    """
    # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L260
    # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L201
    return (
        ExponentialLR(optimizer, 0.8)
        if impl_params
        else DelayedExponentialLR(optimizer, 0.7, 5)
    )
