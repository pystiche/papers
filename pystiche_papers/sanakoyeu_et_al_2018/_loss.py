from abc import abstractmethod
from collections import OrderedDict
from typing import Iterator, Optional, Sequence, Union, cast

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

from pystiche import enc, ops
from pystiche.enc import Encoder, MultiLayerEncoder, SequentialEncoder

from ._modules import (
    DiscriminatorMultiLayerEncoder,
    TransformerBlock,
    prediction_module,
)

__all__ = [
    "EncodingDiscriminatorOperator",
    "PredictionOperator",
    "MultiLayerPredictionOperator",
    "prediction_loss",
    "DiscriminatorLoss",
    "discriminator_loss",
    "transformed_image_loss",
]


class EncodingDiscriminatorOperator(ops.EncodingRegularizationOperator):
    r"""Abstract base class for all discriminator operators working in an encoded space.

    In this operator a discriminator mode can be set, in whose dependence the output can
    be influenced. This can be either ``real`` or ``fake``. In addition, it is
    calculated how accurate the operator was on the last input.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(self, encoder: Encoder, score_weight: float = 1.0):
        super().__init__(encoder, score_weight=score_weight)
        self._target_distribution = True
        self.accuracy: torch.Tensor
        self.register_buffer("accuracy", torch.zeros(1))

    def real(self, mode: bool = True) -> "EncodingDiscriminatorOperator":
        self._target_distribution = mode
        return self

    def fake(self) -> "EncodingDiscriminatorOperator":
        return self.real(False)

    def process_input_image(self, image: torch.Tensor) -> torch.Tensor:
        input_repr = self.input_image_to_repr(image)
        self.accuracy = self.calculate_accuracy(input_repr)
        return self.calculate_score(input_repr)

    @abstractmethod
    def calculate_accuracy(self, input_repr: torch.Tensor) -> torch.Tensor:
        pass


class PredictionOperator(EncodingDiscriminatorOperator):
    r"""The prediction loss is a discriminator loss based on the prediction.

    The prediction consists of the output of an ``predictor`` which processes the
    output of the ``encoder``.

    It measures the cross-entropy loss between true labels and predicted labels. The
    true labels are set depending on the currently set discriminator mode.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        predictor: Auxiliary classifier used to process the encodings into a prediction.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self, encoder: Encoder, predictor: nn.Module, score_weight: float = 1e0,
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
        self.predictor = predictor

    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.predictor(enc))

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            binary_cross_entropy_with_logits(
                input_repr,
                torch.ones_like(input_repr)
                if self._target_distribution
                else torch.zeros_like(input_repr),
            )
        )

    def calculate_accuracy(self, input_repr: torch.Tensor) -> torch.Tensor:
        comparator = torch.ge if self._target_distribution else torch.lt
        return torch.mean(comparator(input_repr, 0.0).float())


class MultiLayerPredictionOperator(ops.MultiLayerEncodingOperator):
    r"""Convenience container for multiple :class:`PredictionOperator` s.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        layers: Layers of the ``multi_layer_encoder`` that the children operators
            operate on.
        get_encoding_op: Callable that returns a children operator given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder`` and its corresponding layer weight.
        layer_weights: Weights of the children operators passed to ``get_encoding_op``.
            If ``"sum"``, each layer weight is set to ``1.0``. If ``"mean"``, each
            layer weight is set to ``1.0 / len(layers)``. If sequence of ``float``s its
            length has to match ``layers``. Defaults to ``"mean"``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def discriminator_operators(self) -> Iterator["EncodingDiscriminatorOperator"]:
        for op in self.operators():
            if isinstance(op, EncodingDiscriminatorOperator):
                yield op

    def real(self, mode: bool = True) -> "MultiLayerPredictionOperator":
        for op in self.discriminator_operators():
            op.real(mode)
        return self

    def fake(self) -> "MultiLayerPredictionOperator":
        return self.real(False)

    def get_accuracy(self) -> torch.Tensor:
        accuracies = torch.stack([op.accuracy for op in self.discriminator_operators()])
        return torch.mean(accuracies)


def prediction_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    scale_weights: Union[str, Sequence[float]] = "sum",
    score_weight: Optional[float] = None,
) -> MultiLayerPredictionOperator:
    r"""Prediction loss indicates whether the input is real or fake.

    Capture image details at different scales with an auxiliary classifier and sum up
    all losses and accuracies on different layers of the
    :class:`~pystiche.enc.MultiLayerEncoder`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        multi_layer_encoder: :class:`~pystiche.enc.MultiLayerEncoder`. If omitted, the
            default
            :class:`~pystiche_papers.sanakoyeu_et_al_2018.DiscriminatorMultiLayerEncoder`
            is used.
        scale_weights: Scale weights of the operator. Defaults to ``sum``.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e0`` if
            ``impl_params is True`` otherwise ``1e-3``.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = DiscriminatorMultiLayerEncoder()

    if score_weight is None:
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L98
        score_weight = 1e0 if impl_params else 1e-3

    predictors = OrderedDict(
        (
            ("0", prediction_module(128, 5)),
            ("1", prediction_module(128, 10)),
            ("3", prediction_module(512, 10)),
            ("5", prediction_module(1024, 6)),
            ("6", prediction_module(1024, 3)),
        )
    )

    def get_encoding_op(
        encoder: enc.SingleLayerEncoder, layer_weight: float
    ) -> PredictionOperator:
        return PredictionOperator(
            encoder, predictors[encoder.layer], score_weight=layer_weight,
        )

    return MultiLayerPredictionOperator(
        multi_layer_encoder,
        tuple(predictors.keys()),
        get_encoding_op,
        layer_weights=scale_weights,
        score_weight=score_weight,
    )


prediction_loss_ = prediction_loss


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

        self.accuracy = torch.mean(torch.cat(accuracies))

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
    r"""Transformed_image_loss from from :cite:`SKL+2018`.

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
