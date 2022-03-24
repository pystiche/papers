from typing import cast, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from pystiche import loss, misc, optim
from pystiche_papers.utils import HyperParameters

from ..utils import batch_up_image
from ._data import images as _images, style_transform as _style_transform
from ._loss import perceptual_loss
from ._modules import transformer as _transformer
from ._utils import (
    hyper_parameters as _hyper_parameters,
    optimizer as _optimizer,
    postprocessor as _postprocessor,
    preprocessor as _preprocessor,
)

__all__ = ["training", "stylization"]


def training(
    content_image_loader: DataLoader,
    style_image: Union[str, torch.Tensor],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    hyper_parameters: Optional[HyperParameters] = None,
    quiet: bool = False,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image: Style image on which the ``transformer`` should be trained. If
            ``str``, the image is read from
            :func:`~pystiche_papers.johnson_alahi_li_2016.images`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. If omitted,
            defaults to ``impl_params``.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
        quiet: If ``True``, not information is logged during the optimization. Defaults
            to ``False``.

    If ``impl_params is True`` , an external preprocessing of the images is used.

    """
    if isinstance(style_image, torch.Tensor):
        device = style_image.device
    else:
        device = misc.get_device()
        images = _images()
        style_image = images[style_image].read(device=device)

    if instance_norm is None:
        instance_norm = impl_params

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transformer = _transformer(impl_params=impl_params, instance_norm=instance_norm)
    transformer = transformer.train().to(device)

    criterion = perceptual_loss(
        impl_params=impl_params, hyper_parameters=hyper_parameters
    )
    criterion = criterion.eval().to(device)

    optimizer = _optimizer(transformer)

    style_transform = _style_transform(hyper_parameters=hyper_parameters)
    style_transform = style_transform.to(device)
    style_image = style_transform(style_image)
    style_image = batch_up_image(style_image, loader=content_image_loader)

    preprocessor = _preprocessor()
    preprocessor = preprocessor.to(device)
    style_image = preprocessor(style_image)

    criterion.set_style_image(style_image)

    def criterion_update_fn(input_image: torch.Tensor, criterion: nn.Module) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(input_image)

    return optim.model_optimization(
        content_image_loader,
        transformer,
        criterion,
        criterion_update_fn,
        optimizer=optimizer,
        quiet=quiet,
    )


def stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    instance_norm: Optional[bool] = None,
    framework: str = "pystiche",
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transfromer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer or the ``style`` to load
            a pretrained transformer with
            :func:`~pystiche_papers.johnson_alahi_li_2016.transformer`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. If omitted,
            defaults to ``impl_params``.
        framework: Framework that was used to train the the transformer. Can be one of
            ``"pystiche"`` (default) and ``"luatorch"``. This only has an effect, if
            if a pretrained ``transformer`` is loaded.
    """
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

    preprocessor = _preprocessor()
    preprocessor = preprocessor.to(device)

    postprocessor = _postprocessor()
    postprocessor = postprocessor.to(device)

    with torch.no_grad():
        input_image = preprocessor(input_image)
        output_image = transformer(input_image)
        output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image).detach()
