from typing import Callable, Optional, Union, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche import loss, misc, optim

from ..utils import batch_up_image
from ._data import content_transform as _content_transform
from ._data import images as _images
from ._data import style_transform as _style_transform
from ._loss import perceptual_loss
from ._modules import transformer as _transformer
from ._utils import lr_scheduler as _lr_scheduler
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["training", "stylization"]


def training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: bool = True,
    transformer: Optional[nn.Module] = None,
    criterion: Optional[loss.PerceptualLoss] = None,
    lr_scheduler: Optional[ExponentialLR] = None,
    num_epochs: Optional[int] = None,
    get_optimizer: Optional[Callable[[nn.Module], Optimizer]] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style: Style image on which the ``transformer`` should be trained. If the
            input is an string, the style image is read from the images in
            :func:`~pystiche_papers.ulyanov_et_al_2016.images`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            :ref:`here <table-branches-ulyanov_et_al_2016>`.
        transformer: Transformer to be optimized. If omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.transformer` is used. Defaults to
            ``None``.
        criterion: Optimization criterion. If omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.perceptual_loss` is used.
            Defaults to ``None``.
        lr_scheduler: LRScheduler. If omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.lr_scheduler` is used. Defaults
            to ``None``.
        num_epochs: Optional number of epochs. If omitted, the num_epochs is determined
            with respect to ``instance_norm`` and ``impl_params``. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        get_optimizer: Optional getter for the optimizer. If omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.optimizer` is used. Defaults to
            ``None``.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    """
    if isinstance(style, str):
        device = misc.get_device()
        images = _images()
        style_image = images[style].read(device=device)
    else:
        style_image = style
        device = style_image.device

    if transformer is None:
        transformer = _transformer(
            impl_params=impl_params, instance_norm=instance_norm,
        )
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = perceptual_loss(
            impl_params=impl_params, instance_norm=instance_norm,
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if lr_scheduler is None:
        if get_optimizer is None:
            get_optimizer = optimizer
        optimizer_ = get_optimizer(transformer)

        lr_scheduler = _lr_scheduler(optimizer_, impl_params=impl_params,)

    if num_epochs is None:
        # The num_iterations are split up into multiple epochs with corresponding
        # num_batches:
        # 50000 = 25 * 2000
        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L48
        # 3000 = 10 * 300
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L30
        # The num_batches is defined in ._data.batch_sampler .
        num_epochs = 25 if impl_params and instance_norm else 10

    style_transform = _style_transform(
        impl_params=impl_params, instance_norm=instance_norm
    )
    style_transform = style_transform.to(device)
    preprocessor = _preprocessor()
    preprocessor = preprocessor.to(device)
    style_image = style_transform(style_image)
    style_image = preprocessor(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)
    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(
            preprocessor(input_image)
        )

    return optim.default_transformer_epoch_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        num_epochs,
        lr_scheduler=lr_scheduler,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: bool = False,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer or string to load a
            pretrained transformer.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            :ref:`here <table-branches-ulyanov_et_al_2016>`.

    """
    device = input_image.device
    if isinstance(transformer, str):
        style = transformer
        transformer = _transformer(
            style=style, impl_params=impl_params, instance_norm=instance_norm,
        )
        if instance_norm or not impl_params:
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/test.lua#L32
            transformer = transformer.eval()
    transformer = transformer.to(device)

    with torch.no_grad():
        content_transform = _content_transform(
            impl_params=impl_params, instance_norm=instance_norm
        )
        content_transform = content_transform.to(device)
        input_image = content_transform(input_image)
        postprocessor = _postprocessor()
        postprocessor = postprocessor.to(device)
        output_image = transformer(input_image)
        output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image.detach())
