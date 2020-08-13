from typing import Callable, Dict, Optional, Tuple, Union

import torch

import pystiche
from pystiche import loss, misc, optim, pyramid

from ._loss import guided_perceptual_loss, perceptual_loss
from ._pyramid import image_pyramid as _image_pyramid
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["nst", "guided_nst"]


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
    r"""NST from :cite:`GEB+2017`.

    Args:
        content_image: Content image for the NST.
        style_image: Style image for the NST.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        criterion: Optimization criterion. If omitted, the default
            :func:`~pystiche_papers.gatys_et_al_2017.perceptual_loss` is used. Defaults
            to ``None``.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    """
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


def guided_nst(
    content_image: torch.Tensor,
    content_guides: Dict[str, torch.Tensor],
    style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    impl_params: bool = True,
    criterion: Optional[loss.GuidedPerceptualLoss] = None,
    image_pyramid: Optional[pyramid.ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    r"""Guided NST from :cite:`GEB+2017`.

    Args:
        content_image: Content image for the guided NST.
        content_guides: Content image guides for the guided NST.
        style_images_and_guides: Dictionary with the style images and the guides.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        criterion: Optimization criterion. If omitted, the default
            :func:`~pystiche_papers.gatys_et_al_2017.guided_perceptual_loss` is used.
            Defaults to ``None``.
        image_pyramid: Image Pyramid. If omitted, the default
            :func:`~pystiche_papers.gatys_et_al_2017.image_pyramid` is used.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    """
    regions = set(content_guides.keys())
    if regions != set(style_images_and_guides.keys()):
        # FIXME
        raise RuntimeError
    regions = sorted(regions)

    if criterion is None:
        criterion = guided_perceptual_loss(regions, impl_params=impl_params)

    if image_pyramid is None:
        image_pyramid = _image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_image_resize = image_pyramid[-1].resize_image
    initial_guide_resize = image_pyramid[-1].resize_guide

    content_image = initial_image_resize(content_image)
    content_guides = {
        region: initial_guide_resize(guide) for region, guide in content_guides.items()
    }
    style_images_and_guides = {
        region: (initial_image_resize(image), initial_guide_resize(guide))
        for region, (image, guide) in style_images_and_guides.items()
    }
    input_image = misc.get_input_image(
        starting_point="content", content_image=content_image
    )

    preprocessor = _preprocessor().to(device)
    postprocessor = _postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))

    for region, (image, guide) in style_images_and_guides.items():
        criterion.set_style_guide(region, guide)
        criterion.set_style_image(region, preprocessor(image))

    for region, guide in content_guides.items():
        criterion.set_content_guide(region, guide)

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
