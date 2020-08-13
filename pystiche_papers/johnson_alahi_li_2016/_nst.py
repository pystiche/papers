from typing import Callable, Optional, Union, cast

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche import loss, misc, optim

from ..utils import batch_up_image
from ._data import images as _images
from ._data import style_transform as _style_transform
from ._loss import perceptual_loss
from ._modules import transformer as _transformer
from ._utils import optimizer as _optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["training", "stylization"]


def training(
    content_image_loader: DataLoader,
    style_image: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    transformer: Optional[nn.Module] = None,
    criterion: Optional[loss.PerceptualLoss] = None,
    optimizer: Optional[Optimizer] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image: Style image on which the ``transformer`` should be trained. If the
            input is an string, the style image is read from the images in
            :func:`~pystiche_papers.johnson_alahi_li_2016._images`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. If omitted,
            defaults to ``impl_params``.
        transformer: Transformer to be optimized. If omitted, the
            :func:`~pystiche_papers.johnson_alahi_li_2016.transformer` is used.
        criterion: Optimization criterion. If omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016.perceptual_loss` is used.
            Defaults to ``None``.
        optimizer: Optimizer. If omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016.optimizer` is used. Defaults
            to ``None``.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If omitted,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If omitted,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    If ``impl_params is True`` , an external preprocessing of the images is used.

    """
    style: Optional[str]
    if isinstance(style_image, torch.Tensor):
        device = style_image.device
        style = None
    else:
        style = style_image
        device = misc.get_device()
        images = _images()
        style_image = images[style_image].read(device=device)

    if instance_norm is None:
        instance_norm = impl_params

    if transformer is None:
        transformer = _transformer(impl_params=impl_params, instance_norm=instance_norm)
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = perceptual_loss(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if optimizer is None:
        optimizer = _optimizer(transformer)

    style_transform = _style_transform(
        impl_params=impl_params, instance_norm=instance_norm, style=style
    )
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    if impl_params:
        # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L111
        # A preprocessor is used in the implementation, which is not documented in the
        # paper.
        preprocessor = _preprocessor()
        preprocessor = preprocessor.to(device)
        style_image = preprocessor(style_image)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(input_image)

    return optim.default_transformer_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    framework: str = "pystiche",
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transfromer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer or string to load a
            pretrained transformer. This string is the style parameter of
            :func:`~pystiche_papers.johnson_alahi_li_2016._transformer`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. If omitted,
            defaults to ``impl_params``.
        framework: Framework that was used to train the the transformer. Can be one of
            ``"pystiche"`` (default) and ``"luatorch"``.
        preprocessor: Optional preprocessor that is called with the ``input_image``
            before the optimization.
        postprocessor: Optional preprocessor that is called with the ``output_image``
            after the optimization.

    If ``impl_params`` is ``True`` , an external preprocessing and postprocessing of the
    images is used.
    """
    device = input_image.device

    if instance_norm is None:
        instance_norm = impl_params

    if isinstance(transformer, str):
        style = transformer
        transformer = _transformer(
            style=style,
            framework=framework,
            impl_params=impl_params,
            instance_norm=instance_norm,
        )
        transformer = transformer.eval()
    transformer = transformer.to(device)

    if impl_params and preprocessor is None:
        # A preprocessor is used in the implementation, which is not documented in the
        # paper.
        # content:
        # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L104
        # style:
        # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L111
        preprocessor = _preprocessor()

    if impl_params and postprocessor is None:
        # A postprocessor is used in the implementation, which is not documented in the
        # paper.
        # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L137
        postprocessor = _postprocessor()

    with torch.no_grad():
        if preprocessor is not None:
            preprocessor = preprocessor.to(device)
            input_image = preprocessor(input_image)

        output_image = transformer(input_image)

        if postprocessor is not None:
            postprocessor = postprocessor.to(device)
            output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image).detach()
