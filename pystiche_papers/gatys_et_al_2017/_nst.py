from typing import Callable, Dict, Optional, Tuple, Union

import torch

import pystiche
from pystiche import misc, optim
from pystiche_papers.utils import HyperParameters

from ._loss import guided_perceptual_loss, perceptual_loss
from ._pyramid import image_pyramid as _image_pyramid
from ._utils import hyper_parameters as _hyper_parameters
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = ["nst", "guided_nst"]


def nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
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
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.
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
        hyper_parameters = _hyper_parameters()

    device = content_image.device

    criterion = perceptual_loss(
        impl_params=impl_params, hyper_parameters=hyper_parameters
    )
    criterion = criterion.to(device)

    image_pyramid = _image_pyramid(
        hyper_parameters=hyper_parameters, resize_targets=(criterion,)
    )
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
    hyper_parameters: Optional[HyperParameters] = None,
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
        style_images_and_guides: Dictionary with the style images and the corresponding
            guides for each region.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.
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

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    device = content_image.device

    criterion = guided_perceptual_loss(
        regions, impl_params=impl_params, hyper_parameters=hyper_parameters
    )
    criterion = criterion.to(device)

    image_pyramid = _image_pyramid(
        hyper_parameters=hyper_parameters, resize_targets=(criterion,)
    )
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
