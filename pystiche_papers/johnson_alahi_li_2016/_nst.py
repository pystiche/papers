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
        preprocessor = _preprocessor()
        """
        https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L111
        """
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
        """
        content:
        https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L104
        style:
        https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L111
        """
        preprocessor = _preprocessor()

    if impl_params and postprocessor is None:
        """
        https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/slow_neural_style.lua#L137
        """
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
