from typing import Callable, Optional, Union

import torch

import pystiche
from pystiche import loss, misc, optim, pyramid

from ._loss import perceptual_loss
from ._pyramid import image_pyramid as _image_pyramid
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["nst"]


def nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[loss.PerceptualLoss] = None,
    image_pyramid: Optional[pyramid.ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = perceptual_loss(impl_params=impl_params)

    if image_pyramid is None:
        image_pyramid = _image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = image_pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = misc.get_input_image(
        starting_point="content", content_image=content_image
    )

    preprocessor = _preprocessor().to(device)
    postprocessor = _postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return optim.default_image_pyramid_optim_loop(
        input_image,
        criterion,
        image_pyramid,
        get_optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )
