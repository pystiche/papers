from typing import Callable, Optional, Union, cast

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pystiche
from pystiche.loss import PerceptualLoss
from pystiche.misc import get_device
from pystiche.optim import OptimLogger, default_transformer_optim_loop

from ..utils import batch_up_image
from .data import (
    johnson_alahi_li_2016_dataset,
    johnson_alahi_li_2016_image_loader,
    johnson_alahi_li_2016_images,
    johnson_alahi_li_2016_style_transform,
)
from .loss import johnson_alahi_li_2016_perceptual_loss
from .modules import johnson_alahi_li_2016_transformer
from .utils import (
    johnson_alahi_li_2016_optimizer,
    johnson_alahi_li_2016_postprocessor,
    johnson_alahi_li_2016_preprocessor,
)

__all__ = [
    "johnson_alahi_li_2016_transformer",
    "johnson_alahi_li_2016_perceptual_loss",
    "johnson_alahi_li_2016_dataset",
    "johnson_alahi_li_2016_image_loader",
    "johnson_alahi_li_2016_training",
    "johnson_alahi_li_2016_images",
    "johnson_alahi_li_2016_stylization",
]


def johnson_alahi_li_2016_training(
    content_image_loader: DataLoader,
    style_image: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    transformer: Optional[nn.Module] = None,
    criterion: Optional[PerceptualLoss] = None,
    optimizer: Optional[Optimizer] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict], float, float], None]
    ] = None,
) -> nn.Module:
    style: Optional[str]
    if isinstance(style_image, torch.Tensor):
        device = style_image.device
        style = None
    else:
        style = style_image
        device = get_device()
        images = johnson_alahi_li_2016_images()
        style_image = images[style_image].read(device=device)

    if instance_norm is None:
        instance_norm = impl_params

    if transformer is None:
        transformer = johnson_alahi_li_2016_transformer(
            impl_params=impl_params, instance_norm=instance_norm
        )
        transformer = transformer.train()
    transformer = transformer.to(device)

    if criterion is None:
        criterion = johnson_alahi_li_2016_perceptual_loss(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        )
        criterion = criterion.eval()
    criterion = criterion.to(device)

    if optimizer is None:
        optimizer = johnson_alahi_li_2016_optimizer(transformer)

    style_transform = johnson_alahi_li_2016_style_transform(
        impl_params=impl_params, instance_norm=instance_norm, style=style
    )
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    if impl_params:
        preprocessor = johnson_alahi_li_2016_preprocessor()
        preprocessor = preprocessor.to(device)
        style_image = preprocessor(style_image)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        cast(PerceptualLoss, criterion).set_content_image(input_image)

    return default_transformer_optim_loop(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def johnson_alahi_li_2016_stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    weights: str = "pystiche",
    preprocessor: Optional[nn.Module] = None,
    postprocessor: Optional[nn.Module] = None,
) -> torch.Tensor:
    device = input_image.device

    if instance_norm is None:
        instance_norm = impl_params

    if isinstance(transformer, str):
        style = transformer
        transformer = johnson_alahi_li_2016_transformer(
            style=style,
            weights=weights,
            impl_params=impl_params,
            instance_norm=instance_norm,
        )
        transformer = transformer.eval()
    transformer = transformer.to(device)

    if impl_params and preprocessor is None:
        preprocessor = johnson_alahi_li_2016_preprocessor()

    if impl_params and postprocessor is None:
        postprocessor = johnson_alahi_li_2016_postprocessor()

    with torch.no_grad():
        if preprocessor is not None:
            preprocessor = preprocessor.to(device)
            input_image = preprocessor(input_image)

        output_image = transformer(input_image)

        if postprocessor is not None:
            postprocessor = postprocessor.to(device)
            output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image).detach()
