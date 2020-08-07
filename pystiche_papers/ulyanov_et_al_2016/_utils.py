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
    r""" Multi-layer encoder based on the VGG19 architecture with the weights of caffe,
    no internal preprocessing and allowed inplace.
    """
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
    r"""

    Args:
        transformer: Transformer to be optimized.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this flag is used for
            switching between the github branches. For details see FIXME.

    Returns:
        :class:`torch.optim.Adam` optimizer with a learning rate of ``1e-3`` if ``impl_params`` and ``instance_norm``
        else ``1e-1``. The parameters of ``transformer`` are set as optimization parameters.

    """
    lr = 1e-3 if impl_params and instance_norm else 1e-1
    return optim.Adam(transformer.parameters(), lr=lr)


class DelayedExponentialLR(ExponentialLR):
    r"""Decays the learning rate of each parameter group by gamma every epoch after a certain number of epochs.
    When last_epoch=-1, sets initial lr as lr.

     Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        delay (int): Number of epochs before the learning rate is reduced with each epoch.
        last_epoch (int): The index of last epoch. Default: -1.

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
    r""" LRScheduler.
    Args:
        optimizer: Wrapped optimizer.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
    """
    return (
        ExponentialLR(optimizer, 0.8)
        if impl_params
        else DelayedExponentialLR(optimizer, 0.7, 5)
    )
