from typing import Any, List, Optional

from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

__all__ = [
    "hyper_parameters",
    "multi_layer_encoder",
    "preprocessor",
    "postprocessor",
    "optimizer",
    "DelayedExponentialLR",
    "lr_scheduler",
]


def hyper_parameters(
    impl_params: bool = True, instance_norm: bool = True
) -> HyperParameters:
    r"""Hyper parameters from :cite:`ULVL2016,UVL2017`."""
    return HyperParameters(
        content_loss=HyperParameters(
            layer="relu4_2",
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L22
            score_weight=6e-1 if impl_params and not instance_norm else 1e0,
        ),
        style_loss=HyperParameters(
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L44
            layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1")
            if impl_params and instance_norm
            else ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
            layer_weights=[1e3]*4 if impl_params and instance_norm else [1e3]*5,
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L23
            score_weight=1e0 if impl_params and not instance_norm else 1e0,
        ),
        content_transform=HyperParameters(edge_size=256,),
        style_transform=HyperParameters(
            edge_size=256,
            # https://github.com/torch/image/blob/master/doc/simpletransform.md#res-imagescalesrc-size-mode
            edge="long",
            interpolation_mode="bicubic"
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L152
            if impl_params and instance_norm
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/descriptor_net.lua#L17
            else "bilinear",
        ),
        batch_sampler=HyperParameters(
            # The number of iterations is split up into multiple epochs with
            # corresponding num_batches:
            num_batches=(
                # 50000 = 25 * 2000
                # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L48
                2000
                if instance_norm
                # 3000 = 10 * 300
                # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L30
                else 300
            )
            if impl_params
            else 200,
            batch_size=(
                (
                    # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L50
                    1
                    if instance_norm
                    # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L32
                    else 4
                )
                if impl_params
                else 16
            ),
        ),
        optimizer=HyperParameters(
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L29
            lr=1e-3
            if impl_params and instance_norm
            else 1e-1,
        ),
        lr_scheduler=HyperParameters(
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L260
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L201
            lr_decay=0.8 if impl_params else 0.7,
            delay=0 if impl_params else 4,
        ),
        # The number of iterations is split up into multiple epochs with
        # corresponding num_batches:
        num_epochs=(
            # 50000 = 25 * 2000
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L48
            25
            if impl_params and instance_norm
            # 3000 = 10 * 300 / 2000 = 10 * 200
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L30
            else 10
        ),
    )


_hyper_parameters = hyper_parameters


def multi_layer_encoder() -> enc.VGGMultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`ULVL2016,UVL2017`."""
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def optimizer(
    transformer: nn.Module,
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> optim.Adam:
    r"""Optimizer from :cite:`ULVL2016,UVL2017`.

    Args:
        transformer: Transformer to be optimized.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )
    return optim.Adam(transformer.parameters(), lr=hyper_parameters.optimizer.lr)


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


def lr_scheduler(
    optimizer: Optimizer,
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ExponentialLR:
    r"""Learning rate scheduler from :cite:`ULVL2016,UVL2017`.

    Args:
        optimizer: Wrapped optimizer.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    return DelayedExponentialLR(
        optimizer,
        hyper_parameters.lr_scheduler.lr_decay,
        hyper_parameters.lr_scheduler.delay,
    )
