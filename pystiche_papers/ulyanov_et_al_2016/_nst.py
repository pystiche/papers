from typing import Callable, Optional, Union, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

import pystiche
from pystiche import loss, misc, optim
from pystiche_papers.utils import HyperParameters

from ..utils import batch_up_image
from ._data import content_transform as _content_transform
from ._data import images as _images
from ._data import style_transform as _style_transform
from ._loss import perceptual_loss
from ._modules import transformer as _transformer
from ._utils import hyper_parameters as _hyper_parameters
from ._utils import lr_scheduler as _lr_scheduler
from ._utils import optimizer as _optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["training", "stylization"]


def training(
    content_image_loader: DataLoader,
    style: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
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
            input is :class:`str`, the style image is read from
            :func:`~pystiche_papers.ulyanov_et_al_2016.images`.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    if isinstance(style, str):
        device = misc.get_device()
        images = _images()
        style_image = images[style].read(device=device)
    else:
        style_image = style
        device = style_image.device

    transformer = _transformer(impl_params=impl_params, instance_norm=instance_norm,)
    transformer = transformer.train()
    transformer = transformer.to(device)

    criterion = perceptual_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        hyper_parameters=hyper_parameters,
    )
    criterion = criterion.eval()
    criterion = criterion.to(device)

    optimizer = _optimizer(transformer)
    lr_scheduler = _lr_scheduler(
        optimizer,
        impl_params=impl_params,
        instance_norm=instance_norm,
        hyper_parameters=hyper_parameters,
    )

    style_transform = _style_transform(
        impl_params=impl_params,
        instance_norm=instance_norm,
        hyper_parameters=hyper_parameters,
    )
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    preprocessor = _preprocessor()
    preprocessor = preprocessor.to(device)
    style_image = preprocessor(style_image)

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
        hyper_parameters.num_epochs,
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
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.

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

    postprocessor = _postprocessor()
    postprocessor = postprocessor.to(device)

    with torch.no_grad():
        content_transform = _content_transform(
            impl_params=impl_params, instance_norm=instance_norm
        )
        content_transform = content_transform.to(device)
        input_image = content_transform(input_image)

        output_image = transformer(input_image)
        output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image.detach())
