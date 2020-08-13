from typing import Callable, Optional, Union

import torch

import pystiche
from pystiche import loss, misc, optim

from ._loss import perceptual_loss
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = [
    "nst",
]


def nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    num_steps: int = 500,
    impl_params: bool = True,
    criterion: Optional[loss.PerceptualLoss] = None,
    quiet: bool = False,
    logger: Optional[optim.OptimLogger] = None,
    log_fn: Optional[
        Callable[[int, Union[torch.Tensor, pystiche.LossDict]], None]
    ] = None,
) -> torch.Tensor:
    r"""NST from :cite:`GEB2016`.

    Args:
        content_image: Content image for the NST.
        style_image: Style image for the NST.
        num_steps: Number of steps for each level.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        criterion: Optimization criterion. If omitted, the default
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.perceptual_loss` is used.
            Defaults to ``None``.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.
        logger: Optional custom logger. If ``None``,
            :class:`pystiche.optim.OptimLogger` is used. Defaults to ``None``.
        log_fn: Optional custom logging function. It is called in every optimization
            step with the current step and loss. If ``None``,
            :func:`~pystiche.optim.default_image_optim_log_fn` is used. Defaults to
            ``None``.

    If ``impl_params is True`` the content_image is set as the starting point instead of
    a random initialized image.
    """
    if criterion is None:
        criterion = perceptual_loss(impl_params=impl_params)

    device = content_image.device
    criterion = criterion.to(device)

    # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
    # Cell [6]
    starting_point = "content" if impl_params else "random"
    input_image = misc.get_input_image(
        starting_point=starting_point, content_image=content_image
    )

    preprocessor = _preprocessor().to(device)
    postprocessor = _postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return optim.default_image_optim_loop(
        input_image,
        criterion,
        get_optimizer=optimizer,
        num_steps=num_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
        log_fn=log_fn,
    )
