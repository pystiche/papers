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
