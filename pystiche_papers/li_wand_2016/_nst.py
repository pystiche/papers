from typing import Callable, Optional, Union

import torch

import pystiche
from pystiche import misc, optim
from pystiche_papers.utils import HyperParameters

from ._loss import perceptual_loss
from ._pyramid import image_pyramid as _image_pyramid
from ._utils import (
    hyper_parameters as _hyper_parameters,
    optimizer,
    postprocessor as _postprocessor,
    preprocessor as _preprocessor,
)

__all__ = ["nst"]


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
    r"""NST from :cite:`LW2016`.

    Args:
        content_image: Content image for the NST.
        style_image: Style image for the NST.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.
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
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

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
        starting_point=hyper_parameters.nst.starting_point, content_image=content_image
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
