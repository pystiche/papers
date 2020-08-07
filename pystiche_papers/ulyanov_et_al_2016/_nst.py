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
        style: Style image on which the ``transformer`` should be trained.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this flag is used for
            switching between the github branches. For details see FIXME.
        transformer: Transformer to be optimized. If ``None``, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.transformer` from the paper is used.
        criterion: Optimization criterion. If ``None``, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.perceptual_loss` from the paper is used.
            Defaults to ``None``.
        lr_scheduler: LRScheduler. If ``None``, the default :func:`~pystiche_papers.ulyanov_et_al_2016.lr_scheduler`
            from the paper is used. Defaults to ``None``.
        num_epochs: Optional number of epochs. If ``omitted``, the num_epochs is determined with respect to
            ``instance_norm`` and ``impl_params``. For details see FIXME
        get_optimizer: Optional getter for the optimizer. If ``None``,
            :func:`~pystiche_papers.ulyanov_et_al_2016.optimizer` is used. Defaults to ``None``
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
    r"""Transforms an input image into a stylised version using the transfromer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer or string to load a pretrained transformer.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this flag is used for
            switching between the github branches. For details see FIXME.

    """
    device = input_image.device
    if isinstance(transformer, str):
        style = transformer
        transformer = _transformer(
            style=style, impl_params=impl_params, instance_norm=instance_norm,
        )
        if instance_norm or not impl_params:
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
