from typing import Callable, Dict, Optional, Tuple, Union

import torch

import pystiche
from pystiche.loss import GuidedPerceptualLoss, PerceptualLoss
from pystiche.misc import get_input_image
from pystiche.optim import OptimLogger, default_image_pyramid_optim_loop
from pystiche.pyramid import ImagePyramid

from ._loss import guided_perceptual_loss, perceptual_loss
from ._pyramid import image_pyramid
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["nst", "guided_nst"]


def nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[PerceptualLoss] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = perceptual_loss(impl_params=impl_params)

    if pyramid is None:
        pyramid = image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = _preprocessor().to(device)
    postprocessor = _postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )


def guided_nst(
    content_image: torch.Tensor,
    content_guides: Dict[str, torch.Tensor],
    style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    impl_params: bool = True,
    criterion: Optional[GuidedPerceptualLoss] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    regions = set(content_guides.keys())
    if regions != set(style_images_and_guides.keys()):
        # FIXME
        raise RuntimeError
    regions = sorted(regions)

    if criterion is None:
        criterion = guided_perceptual_loss(regions, impl_params=impl_params)

    if pyramid is None:
        pyramid = image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_image_resize = pyramid[-1].resize_image
    initial_guide_resize = pyramid[-1].resize_guide

    content_image = initial_image_resize(content_image)
    content_guides = {
        region: initial_guide_resize(guide) for region, guide in content_guides.items()
    }
    style_images_and_guides = {
        region: (initial_image_resize(image), initial_guide_resize(guide))
        for region, (image, guide) in style_images_and_guides.items()
    }
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = _preprocessor().to(device)
    postprocessor = _postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))

    for region, (image, guide) in style_images_and_guides.items():
        criterion.set_style_guide(region, guide)
        criterion.set_style_image(region, preprocessor(image))

    for region, guide in content_guides.items():
        criterion.set_content_guide(region, guide)

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )
