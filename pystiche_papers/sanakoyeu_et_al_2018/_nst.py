from typing import Callable, Optional, Tuple, Union, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pystiche import loss, misc
from pystiche.image.transforms import functional as F

from ._loss import (
    DiscriminatorLoss,
    MultiLayerPredictionOperator,
    prediction_loss,
    transformer_loss,
)
from ._transformer import transformer as _transformer
from ._utils import ExponentialMovingAverageMeter
from ._utils import lr_scheduler as _lr_scheduler
from ._utils import optimizer
from ._utils import postprocessor as _postprocessor
from ._utils import preprocessor as _preprocessor

__all__ = [
    "gan_optim_loop",
    "gan_epoch_optim_loop",
    "training",
    "stylization",
]


def gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    discriminator_criterion: DiscriminatorLoss,
    transformer_criterion: loss.PerceptualLoss,
    transformer_criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    target_win_rate: float = 0.8,
    impl_params: bool = True,
) -> nn.Module:
    r"""Perform a GAN optimization for a single epoch.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image_loader: Style images used as input for the ``discriminator``.
        transformer: Transformer to be optimized.
        discriminator_criterion: Optimization criterion for the ``discriminator``.
        transformer_criterion: Optimization criterion for the ``transformer``.
        transformer_criterion_update_fn: Is called before each optimization step with
            the current images and the optimization ``transformer_criterion``.
        discriminator_optimizer: Optional optimizer for the ``discriminator``. If
            ``None``, it is extracted from ``discriminator_lr_scheduler`` or the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.optimizer` is used.
        transformer_optimizer: Optional optimizer for the ``transformer``. If
            ``None``, it is extracted from ``transformer_lr_scheduler`` or the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.optimizer` is used.
        target_win_rate: Initial value for the success of the discriminator, which also
            serves as a limit for the alternate training of the transformer and the
            discriminator. If the ``discriminator_success < target_win_rate``, the
            ``discriminator`` is updated and the ``transformer`` otherwise.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    If ``impl_params is True``, then in addition to the stylized images and the style
    images, the content images used are also included in the loss and accuracy
    calculation.

    """
    device = misc.get_device()
    style_image_loader = iter(style_image_loader)
    preprocessor = _preprocessor()

    if discriminator_optimizer is None:
        discriminator_optimizer = optimizer(
            discriminator_criterion.prediction_loss.parameters()
        )

    if transformer_optimizer is None:
        transformer_optimizer = optimizer(transformer)

    if "discriminator_success" not in locals():
        discriminator_success = ExponentialMovingAverageMeter(
            "discriminator_success", init_val=target_win_rate
        )

    def train_discriminator_one_step(
        output_image: torch.Tensor,
        style_image: torch.Tensor,
        input_image: Optional[torch.Tensor] = None,
    ) -> None:
        def closure() -> float:
            cast(Optimizer, discriminator_optimizer).zero_grad()
            loss = discriminator_criterion(output_image, style_image, input_image)
            loss.backward()
            return cast(float, loss.item())

        cast(Optimizer, discriminator_optimizer).step(closure)
        discriminator_success.update(discriminator_criterion.accuracy)

    def train_transformer_one_step(output_image: torch.Tensor) -> None:
        def closure() -> float:
            cast(Optimizer, transformer_optimizer).zero_grad()
            cast(MultiLayerPredictionOperator, transformer_criterion.style_loss).real()
            loss = transformer_criterion(output_image)
            loss.backward()
            return cast(float, loss.item())

        cast(Optimizer, transformer_optimizer).step(closure)
        accuracy = cast(
            MultiLayerPredictionOperator, transformer_criterion.style_loss
        ).get_accuracy()
        discriminator_success.update(1.0 - accuracy)

    for content_image in content_image_loader:
        input_image = content_image.to(device)

        output_image = transformer(preprocessor(input_image))

        if discriminator_success.local_avg < target_win_rate:
            style_image = next(style_image_loader)
            style_image = style_image.to(device)
            style_image = preprocessor(style_image)
            train_discriminator_one_step(
                output_image,
                style_image,
                input_image=input_image if impl_params else None,
            )
        else:
            transformer_criterion_update_fn(input_image, transformer_criterion)
            train_transformer_one_step(output_image)

    return transformer


def gan_epoch_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    epochs: int,
    discriminator_criterion: DiscriminatorLoss,
    transformer_criterion: loss.PerceptualLoss,
    transformer_criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    discriminator_lr_scheduler: Optional[LRScheduler] = None,
    transformer_lr_scheduler: Optional[LRScheduler] = None,
    target_win_rate: float = 0.8,
    impl_params: bool = True,
) -> nn.Module:
    r"""Perform a GAN optimization for multiple epochs.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image_loader: Style images used as input for the ``discriminator``.
        transformer: Transformer to be optimized.
        epochs: Number of epochs.
        discriminator_criterion: Optimization criterion for the ``discriminator``.
        transformer_criterion: Optimization criterion for the ``transformer``.
        transformer_criterion_update_fn: Is called before each optimization step with
            the current images and the optimization ``transformer_criterion``.
        discriminator_optimizer: Optional optimizer for the ``discriminator``. If
            ``None``, it is extracted from ``discriminator_lr_scheduler`` or the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.optimizer` is used.
        transformer_optimizer: Optional optimizer for the ``transformer``. If
            ``None``, it is extracted from ``transformer_lr_scheduler`` or the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.optimizer` is used.
        discriminator_lr_scheduler: LRScheduler for the ``discriminator``. If omitted,
            the default :func:`~pystiche_papers.sanakoyeu_et_al_2018.lr_scheduler` is
            used.
        transformer_lr_scheduler: LRScheduler for the ``transformer``. If omitted, the
            default :func:`~pystiche_papers.sanakoyeu_et_al_2018.lr_scheduler` is
            used.
        target_win_rate: Initial value for the success of the discriminator, which also
            serves as a limit for the alternate training of the transformer and the
            discriminator. If the ``discriminator_success < target_win_rate``, the
            ``discriminator`` is updated and the ``transformer`` otherwise.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    """
    if discriminator_optimizer is None:
        if discriminator_lr_scheduler is None:
            discriminator_optimizer = optimizer(
                discriminator_criterion.prediction_loss.parameters()
            )
        else:
            discriminator_optimizer = discriminator_lr_scheduler.optimizer  # type: ignore[attr-defined]

    if transformer_optimizer is None:
        if transformer_lr_scheduler is None:
            transformer_optimizer = optimizer(transformer)
        else:
            transformer_optimizer = transformer_lr_scheduler.optimizer  # type: ignore[attr-defined]

    def optim_loop(transformer: nn.Module) -> nn.Module:
        return gan_optim_loop(
            content_image_loader,
            style_image_loader,
            transformer,
            discriminator_criterion,
            transformer_criterion,
            transformer_criterion_update_fn,
            discriminator_optimizer=discriminator_optimizer,
            transformer_optimizer=transformer_optimizer,
            target_win_rate=target_win_rate,
            impl_params=impl_params,
        )

    for epoch in range(epochs):
        transformer = optim_loop(transformer)

        if discriminator_lr_scheduler is not None:
            discriminator_lr_scheduler.step(epoch)

        if transformer_lr_scheduler is not None:
            transformer_lr_scheduler.step(epoch)

    return transformer


def training(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    impl_params: bool = True,
) -> nn.Module:
    r"""Training a transformer for the NST.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
        style_image_loader: Style images used as input for the ``discriminator``.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    """
    device = misc.get_device()

    transformer = _transformer()
    transformer = transformer.train()
    transformer = transformer.to(device)

    prediction_operator = prediction_loss(impl_params=impl_params)

    discriminator_criterion = DiscriminatorLoss(prediction_operator)
    discriminator_criterion = discriminator_criterion.eval()
    discriminator_criterion = discriminator_criterion.to(device)

    transformer_criterion = transformer_loss(
        transformer.encoder, impl_params=impl_params
    )
    transformer_criterion = transformer_criterion.eval()
    transformer_criterion = transformer_criterion.to(device)

    get_optimizer = optimizer

    def transformer_criterion_update_fn(
        content_image: torch.Tensor, criterion: nn.Module
    ) -> None:
        cast(loss.PerceptualLoss, criterion).set_content_image(content_image)

    discriminator_optimizer = get_optimizer(prediction_operator.parameters())
    discriminator_lr_scheduler = _lr_scheduler(discriminator_optimizer)

    transformer_optimizer = get_optimizer(transformer.parameters())
    transformer_lr_scheduler = _lr_scheduler(transformer_optimizer)

    # The num_iterations are split up into multiple epochs with corresponding
    # num_batches:
    # The number of epochs is defined in _data.batch_sampler.
    # 300_000 = 1 * 300_000
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L68
    # 300_000 = 3 * 100_000
    num_epochs = 1 if impl_params else 3

    return gan_epoch_optim_loop(
        content_image_loader,
        style_image_loader,
        transformer,
        num_epochs,
        discriminator_criterion,
        transformer_criterion,
        transformer_criterion_update_fn,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        transformer_lr_scheduler=transformer_lr_scheduler,
        impl_params=impl_params,
    )


def stylization(
    input_image: torch.Tensor,
    transformer: Union[nn.Module, str],
    impl_params: bool = True,
    transform_size: Union[int, Tuple[int, int]] = 768,
) -> torch.Tensor:
    r"""Transforms an input image into a stylised version using the transformer.

    Args:
        input_image: Image to be stylised.
        transformer: Pretrained transformer for style transfer or string to load a
            pretrained ``transformer``.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        transform_size: Size to which the image is resized before transforming with the
            ``transformer``. If :class:`int` is given, the size refers to the smaller
            edge. Default to ``768``.

    """
    device = input_image.device

    preprocessor = _preprocessor()
    postprocessor = _postprocessor()
    if isinstance(transformer, str):
        style = transformer
        transformer = _transformer(style=style)

    if not impl_params:
        transformer = transformer.eval()
    transformer = transformer.to(device)

    with torch.no_grad():
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L492-L495'
        input_image = F.resize(input_image, transform_size, edge="short")
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L500
        output_image = transformer(preprocessor(input_image))
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L504
        output_image = postprocessor(output_image)

    return cast(torch.Tensor, output_image.detach())
