

from ._utils import ExponentialMovingAverageMeter
from ._utils import optimizer
from ._utils import preprocessor as _preprocessor

__all__ = [
    "gan_optim_loop",
    "epoch_gan_optim_loop",
]


def gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    discriminator_criterion: nn.Module,
    transformer_criterion: nn.Module,
    transformer_criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    target_win_rate: float = 0.8,
    impl_params: bool = True,
) -> nn.Module:
    r"""Perform a GAN optimization for a single epoch with integrated logging.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
            Drawing from this should yield a single item.
        style_image_loader: Style images used as input for the ``discriminator``.
            Drawing from this should yield a single item.
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

    """
    device = misc.get_device()
    if isinstance(style_image_loader, DataLoader):
        style_image_loader = itertools.cycle(style_image_loader)

    if discriminator_optimizer is None:
        discriminator_optimizer = optimizer(
            discriminator_criterion.discriminator.parameters()
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
        discriminator_success.update(discriminator_criterion.get_current_acc())

    def train_transformer_one_step(output_image: torch.Tensor) -> None:
        def closure() -> float:
            cast(Optimizer, transformer_optimizer).zero_grad()
            loss = transformer_criterion(output_image)
            loss.backward()
            return cast(float, loss.item())

        cast(Optimizer, transformer_optimizer).step(closure)
        discriminator_success.update(
            (1.0 - transformer_criterion.style_loss.get_discriminator_acc())
        )

    for content_image in content_image_loader:
        input_image = content_image.to(device)

        output_image = transformer(_preprocessor(input_image))

        if discriminator_success.local_avg < target_win_rate:
            style_image = next(style_image_loader)
            style_image = style_image.to(device)
            style_image = _preprocessor(style_image)
            train_discriminator_one_step(
                output_image,
                style_image,
                input_image=input_image if impl_params else None,
            )
        else:
            transformer_criterion_update_fn(input_image, transformer_criterion)
            train_transformer_one_step(output_image)

    return transformer


def epoch_gan_optim_loop(
    content_image_loader: DataLoader,
    style_image_loader: DataLoader,
    transformer: nn.Module,
    epochs: int,
    discriminator_criterion: nn.Module,
    transformer_criterion: nn.Module,
    transformer_criterion_update_fn: Callable[[torch.Tensor, nn.Module], None],
    discriminator_optimizer: Optional[Optimizer] = None,
    transformer_optimizer: Optional[Optimizer] = None,
    discriminator_lr_scheduler: Optional[LRScheduler] = None,
    transformer_lr_scheduler: Optional[LRScheduler] = None,
    target_win_rate: float = 0.8,
    impl_params: bool = True,
) -> nn.Module:
    r"""Perform a GAN optimization for multiple epochs with integrated logging.

    Args:
        content_image_loader: Content images used as input for the ``transformer``.
            Drawing from this should yield a single item.
        style_image_loader: Style images used as input for the ``discriminator``.
            Drawing from this should yield a single item.
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
    style_image_loader = itertools.cycle(style_image_loader)

    if discriminator_optimizer is None:
        if discriminator_lr_scheduler is None:
            discriminator_optimizer = optimizer(
                discriminator_criterion.discriminator.get_parameters()
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