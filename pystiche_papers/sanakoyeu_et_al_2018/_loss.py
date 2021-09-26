from typing import List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn

from pystiche import enc, loss, ops
from pystiche_papers.utils import AutoPadAvgPool2d, AutoPadConv2d

from ._discriminator import MultiScaleDiscriminator
from ._discriminator import discriminator as _discriminator
from ._transformer import Transformer
from ._transformer import transformer as _transformer

__all__ = [
    "DiscriminatorLoss",
    "discriminator_loss",
    "MAEFeatureReconstructionOperator",
    "feature_reconstruction_loss",
    "transformed_image_loss",
    "content_loss",
    "StyleLoss",
    "style_loss",
    "transformer_loss",
]


def _prediction_loss(
    discriminator: MultiScaleDiscriminator,
    real_inputs: Optional[Sequence[torch.Tensor]] = None,
    fake_inputs: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    if not (real_inputs or fake_inputs):
        raise RuntimeError
    elif real_inputs is None:
        real_inputs = ()
    elif fake_inputs is None:
        fake_inputs = ()

    def compute(input: torch.Tensor, real: bool) -> torch.Tensor:
        make_target = torch.ones_like if real else torch.zeros_like
        scale_losses = [
            F.binary_cross_entropy_with_logits(prediction, make_target(prediction))
            for prediction in discriminator(input).values()
        ]
        return torch.sum(torch.stack(scale_losses))

    losses: List[torch.Tensor] = []

    with discriminator.record_accuracies() as (real, fake):
        with real():
            for input in cast(Sequence, real_inputs):
                losses.append(compute(input, real=True))

        with fake():
            for input in cast(Sequence, fake_inputs):
                losses.append(compute(input, real=False))

    return torch.sum(torch.stack(losses))


class DiscriminatorLoss(nn.Module):
    def __init__(self, discriminator: MultiScaleDiscriminator) -> None:
        super().__init__()
        self.discriminator = discriminator

    def forward(
        self,
        output_photo: torch.Tensor,
        input_painting: torch.Tensor,
        input_photo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        real_inputs = [input_painting]
        fake_inputs = [output_photo]
        if input_photo is not None:
            fake_inputs.append(input_photo)
        return _prediction_loss(
            cast(MultiScaleDiscriminator, self.discriminator), real_inputs, fake_inputs
        )


def discriminator_loss(
    discriminator: Optional[MultiScaleDiscriminator] = None,
) -> DiscriminatorLoss:
    """Discriminator_loss from :cite:`SKL+2018`.

    Args:
        discriminator: Discriminator module to discriminate between real and fake
            inputs. If omitted, the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.discriminator()` is used.

    """
    if discriminator is None:
        discriminator = _discriminator()
    return DiscriminatorLoss(discriminator=discriminator)


class MAEFeatureReconstructionOperator(ops.EncodingComparisonOperator):
    r"""The MAE feature reconstruction loss is a content loss.

    It measures the mean absolute error (MAE) between the encodings of an
    ``input_image`` :math:`\hat{I}` and a ``target_image`` :math:`I` :

    .. math::

        \mean \abs{\Phi\of{\hat{I}} - \Phi\of{I}}

    Here :math:`\Phi\of{\cdot}` denotes the ``encoder``.

    Args:
        encoder: Encoder :math:`\Phi`.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return enc

    def input_enc_to_repr(
        self, enc: torch.Tensor, ctx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.enc_to_repr(enc)

    def target_enc_to_repr(self, enc: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.enc_to_repr(enc), None

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.l1_loss(input_repr, target_repr)


def feature_reconstruction_loss(
    encoder: enc.Encoder,
    impl_params: bool = True,
    score_weight: Optional[float] = None,
) -> Union[MAEFeatureReconstructionOperator, ops.FeatureReconstructionOperator]:
    r"""Feature reconstruction from :cite:`SKL+2018`.

    Args:
        encoder: :class:`~pystiche.enc.SequentialEncoder`.
        impl_params:  If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e2`` if
            ``impl_params is True`` otherwise ``1e0``.

    If ``impl_params is True``, the ``score`` is calculated with a normalized absolute
    distance instead of a normalized squared euclidean distance.

    """
    if score_weight is None:
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L108
        score_weight = 1e2 if impl_params else 1e0

    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L194
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/module.py#L177-L178
    cls = (
        MAEFeatureReconstructionOperator
        if impl_params
        else ops.FeatureReconstructionOperator
    )
    return cls(encoder, score_weight=score_weight)


# TODO: since this is only used in the transformed image loss, shouldn't it be defined
#  there?
class TransformerBlock(enc.SequentialEncoder):
    r"""TransformerBlock from :cite:`SKL+2018`.

    This block takes an image as input and produce a transformed image of the same size.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
        kernel_size: Size of the convolving kernel. Defaults to ``10``.
        stride: Stride of the convolution. Defaults to ``1``.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    If ``impl_params is True``, an :class:`~torch.nn.AvgPool2d` is used instead of a
    :class:`~torch.nn.Conv2d` with :func:`~torch.nn.utils.weight_norm`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: Union[Tuple[int, int], int] = 10,
        stride: Union[Tuple[int, int], int] = 1,
        impl_params: bool = True,
    ):
        kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/module.py#L246
        module = (
            AutoPadAvgPool2d(**kwargs, count_include_pad=False)
            if impl_params
            else nn.utils.weight_norm(AutoPadConv2d(in_channels, in_channels, **kwargs))
        )
        self.impl_params = impl_params
        super().__init__((module,))


def transformed_image_loss(
    impl_params: bool = True,
    transformer_block: Optional[enc.Encoder] = None,
    score_weight: Optional[float] = None,
) -> ops.FeatureReconstructionOperator:
    r"""Transformed_image_loss from :cite:`SKL+2018`.

    Args:
        transformer_block:
            :class:`~pystiche_papers.sanakoyeu_et_al_2018.TransformerBlock` which is
            used to transform the image.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e2`` if
            ``impl_params is True`` otherwise ``1e0``.

    """
    if transformer_block is None:
        transformer_block = TransformerBlock()

    if score_weight is None:
        score_weight = 1e2 if impl_params else 1e0

    return ops.FeatureReconstructionOperator(
        transformer_block, score_weight=score_weight
    )


def content_loss(
    impl_params: bool = True, transformer: Optional[Transformer] = None
) -> ops.OperatorContainer:
    if transformer is None:
        transformer = _transformer(impl_params=impl_params)
    return ops.OperatorContainer(
        (
            (
                (
                    "feature_reconstruction_loss",
                    feature_reconstruction_loss(transformer.encoder),
                ),
                ("tranformed_image_loss", transformed_image_loss()),
            )
        )
    )


class StyleLoss(ops.PixelRegularizationOperator):
    def __init__(
        self, discriminator: MultiScaleDiscriminator, score_weight: float = 1e0,
    ) -> None:
        super().__init__(score_weight=score_weight)
        self.discriminator = discriminator

    def input_image_to_repr(self, image: torch.Tensor,) -> torch.Tensor:
        return image

    def calculate_score(self, input_repr: torch.Tensor,) -> torch.Tensor:
        return _prediction_loss(self.discriminator, real_inputs=[input_repr])


def style_loss(discriminator: Optional[MultiScaleDiscriminator] = None) -> StyleLoss:
    """Generator_loss from :cite:`SKL+2018`.

    The loss and accuracy of the ``discriminator`` is calculated if the input images are
    to be recognized as real.

    Args:
        discriminator: Discriminator module to discriminate between real and fake
            inputs. If omitted, the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.discriminator()` is used.
    """
    if discriminator is None:
        discriminator = _discriminator()
    return StyleLoss(discriminator)


def transformer_loss(
    impl_params: bool = True,
    discriminator: Optional[MultiScaleDiscriminator] = None,
    transformer: Optional[Transformer] = None,
) -> loss.PerceptualLoss:
    r"""Transformer_loss from :cite:`SKL+2018`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    """
    if discriminator is None:
        discriminator = _discriminator()
    if transformer is None:
        transformer = _transformer(impl_params=impl_params)

    return loss.PerceptualLoss(
        content_loss(transformer=transformer), style_loss(discriminator=discriminator)
    )
