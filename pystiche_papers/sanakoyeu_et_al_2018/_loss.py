from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pystiche import loss, ops
from pystiche.enc import SequentialEncoder

from ._discriminator import MultiLayerPredictionOperator, prediction_loss
from ._modules import TransformerBlock

__all__ = [
    "DiscriminatorLoss",
    "discriminator_loss",
    "transformed_image_loss",
    "MAEReconstructionOperator",
    "style_aware_content_loss",
    "transformer_loss",
]


style_loss_ = prediction_loss_ = prediction_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, prediction_loss: MultiLayerPredictionOperator) -> None:
        super().__init__()
        self.prediction_loss = prediction_loss

        self.accuracy: torch.Tensor
        self.register_buffer("accuracy", torch.zeros(1))

    def forward(
        self,
        output_photo: torch.Tensor,
        input_painting: torch.Tensor,
        input_photo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        accuracies = []

        self.prediction_loss.real()
        loss = self.prediction_loss(input_painting).aggregate(0)
        accuracies.append(self.prediction_loss.get_accuracy())

        self.prediction_loss.fake()
        loss += self.prediction_loss(output_photo).aggregate(0)
        accuracies.append(self.prediction_loss.get_accuracy())

        if input_photo is not None:
            loss += self.prediction_loss(input_photo).aggregate(0)
            accuracies.append(self.prediction_loss.get_accuracy())

        self.accuracy = torch.mean(torch.stack(accuracies))

        return cast(torch.Tensor, loss)


def discriminator_loss(
    impl_params: bool = True,
    prediction_loss: Optional[MultiLayerPredictionOperator] = None,
) -> DiscriminatorLoss:
    r"""Discriminator loss from :cite:`SKL+2018`.

    Calculates the loss and accuracy of the current discriminator on all real and fake
    input images.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        prediction_loss: Trainable :class:`MultiLayerPredictionOperator`.

    """
    if prediction_loss is None:
        prediction_loss = prediction_loss_(impl_params=impl_params)
    return DiscriminatorLoss(prediction_loss)


def transformed_image_loss(
    transformer_block: Optional[SequentialEncoder] = None,
    impl_params: bool = True,
    score_weight: Optional[float] = None,
) -> ops.FeatureReconstructionOperator:
    r"""Transformed_image_loss from :cite:`SKL+2018`.

    Args:
        transformer_block::class:`~pystiche_papers.sanakoyeu_et_al_2018.TransformerBlock`
            which is used to transform the image.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e2`` if
            ``impl_params is True`` otherwise ``1e0``.
    """
    if score_weight is None:
        score_weight = 1e2 if impl_params else 1e0

    if transformer_block is None:
        transformer_block = TransformerBlock()

    return ops.FeatureReconstructionOperator(
        transformer_block, score_weight=score_weight
    )


class MAEReconstructionOperator(ops.EncodingComparisonOperator):
    r"""The MAE reconstruction loss is a content loss.

    It measures the mean absolute error (MAE) between the encodings of an
    ``input_image`` :math:`\hat{I}` and a ``target_image`` :math:`I` :

    .. math::

        \mean |\parentheses{\Phi\of{\hat{I}} - \Phi\of{I}}|

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


def style_aware_content_loss(
    encoder: SequentialEncoder,
    impl_params: bool = True,
    score_weight: Optional[float] = None,
) -> Union[MAEReconstructionOperator, ops.FeatureReconstructionOperator]:
    r"""Style_aware_content_loss from :cite:`SKL+2018`.

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
    return (
        MAEReconstructionOperator(encoder, score_weight=score_weight)
        if impl_params
        else ops.FeatureReconstructionOperator(encoder, score_weight=score_weight)
    )


def transformer_loss(
    encoder: SequentialEncoder,
    prediction_loss: Optional[MultiLayerPredictionOperator] = None,
    impl_params: bool = True,
    style_aware_content_kwargs: Optional[Dict[str, Any]] = None,
    transformed_image_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    r"""Transformer_loss from :cite:`SKL+2018`.

    Args:
        encoder: :class:`~pystiche.enc.SequentialEncoder`.
        prediction_loss: Trainable :class:`MultiLayerPredictionOperator`.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        style_aware_content_kwargs: Optional parameters for the
            :func:`style_aware_content_loss`.
        transformed_image_kwargs: Optional parameters for the
            :func:`transformed_image_loss`.

    """
    if style_aware_content_kwargs is None:
        style_aware_content_kwargs = {}
    style_aware_content_operator = style_aware_content_loss(
        encoder, impl_params=impl_params, **style_aware_content_kwargs
    )

    if transformed_image_kwargs is None:
        transformed_image_kwargs = {}
    transformed_image_operator = transformed_image_loss(
        impl_params=impl_params, **transformed_image_kwargs
    )

    content_loss = ops.OperatorContainer(
        (
            (
                ("style_aware_content_loss", style_aware_content_operator),
                ("tranformed_image_loss", transformed_image_operator),
            )
        )
    )

    style_loss = cast(ops.OperatorContainer, prediction_loss)
    return loss.PerceptualLoss(content_loss, style_loss)
