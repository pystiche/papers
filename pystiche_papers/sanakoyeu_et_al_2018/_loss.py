from typing import Optional

from pystiche.enc.encoder import SequentialEncoder
from pystiche.ops.comparison import FeatureReconstructionOperator

from ._modules import TransformerBlock

__all__ = ["transformed_image_loss"]


def transformed_image_loss(
    transformer_block: Optional[SequentialEncoder] = None,
    impl_params: bool = True,
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
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

    return FeatureReconstructionOperator(transformer_block, score_weight=score_weight)

from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import ops
from pystiche.enc import Encoder, MultiLayerEncoder

__all__ = [
    "DiscriminatorEncodingOperator",
    "MultiLayerDicriminatorEncodingOperator",
    "discriminator_operator",
    "DiscriminatorLoss",
]


class DiscriminatorEncodingOperator(ops.EncodingRegularizationOperator):
    r"""Discriminator encoding operator working in an encoded space.

    In addition to the ``loss``, this operator also determines the current accuracy of
    the discriminator depending on the parameter ``real``.

    Args:
        encoder: Encoder that is used to encode the target and input images.
        prediction_module: Auxiliary classifier used to process the encodings into a
            prediction.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self, encoder: Encoder, prediction_module: nn.Module, score_weight: float = 1e0,
    ) -> None:
        super().__init__(encoder, score_weight=score_weight)
        self.prediction_module = prediction_module
        self.acc = torch.empty(1)

    def input_enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.prediction_module(enc))

    def process_input_image(
        self, image: torch.Tensor, real: bool = True
    ) -> torch.Tensor:
        return self.calculate_score(self.input_image_to_repr(image), real=real)

    def _loss(self, prediction: torch.Tensor, real: bool) -> torch.Tensor:
        r"""Calculation of the loss.

        The loss calculation is performed using a
        :func:`~torch.nn.functional.binary_cross_entropy_with_logits` on the prediction
        of the operator. The target ``logits`` are ``1`` if it is ``real`` and ``0``
        otherwise.

        Args:
            prediction: Prediction of the input.
            real: If ``True``, the ``prediction`` should be classified as ``real``,
                otherwise as a fake.

        """
        return binary_cross_entropy_with_logits(
            prediction,
            torch.ones_like(prediction) if real else torch.zeros_like(prediction),
        )

    def _acc(self, prediction: torch.Tensor, real: bool) -> torch.Tensor:
        r"""Calculation of the accuracy.

        The accuracy calculation corresponds to the proportion of entries in the
        ``prediction`` that are greater than ``0`` if it is ``real`` otherwise less than
        ``0``.

        Args:
            prediction: Prediction of the input.
            real: If ``True``, the ``prediction`` should be classified as ``real``,
                otherwise as a fake.

        """

        def get_acc_mask(prediction: torch.Tensor, real: bool) -> torch.Tensor:
            if real:
                return torch.masked_fill(
                    torch.zeros_like(prediction),
                    prediction > torch.zeros_like(prediction),
                    1,
                )
            else:
                return torch.masked_fill(
                    torch.zeros_like(prediction),
                    prediction < torch.zeros_like(prediction),
                    1,
                )

        return torch.mean(get_acc_mask(prediction, real))

    def get_current_acc(self) -> torch.Tensor:
        return self.acc

    def calculate_score(
        self, prediction: torch.Tensor, real: bool = True
    ) -> torch.Tensor:
        self.acc = self._acc(prediction, real)
        return self._loss(prediction, real=real)

    def forward(
        self, input_image: torch.Tensor, real: bool = True
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image, real=real) * self.score_weight


class MultiLayerDicriminatorEncodingOperator(ops.MultiLayerEncodingOperator):
    r"""Convenience container for multiple :class:`DiscriminatorEncodingOperator` s.

    Args:
        encoder: :class:`~pystiche.enc.MultiLayerEncoder`.
        layers: Layers of the ``encoder`` that the children operators operate on.
        get_encoding_op: Callable that returns a children operator given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``encoder`` and its corresponding layer weight.
        layer_weights: Weights of the children operators passed to ``get_encoding_op``.
            If ``"sum"``, each layer weight is set to ``1.0``. If ``"mean"``, each
            layer weight is set to ``1.0 / len(layers)``. If sequence of ``float``s its
            length has to match ``layers``. Defaults to ``"sum"``.
        score_weight: Score weight of the operator. Defaults to ``1.0``.
    """

    def __init__(
        self,
        encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], ops.EncodingOperator],
        layer_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e0,
    ):
        super().__init__(
            encoder,
            layers,
            get_encoding_op,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )
        self.encoder_parameters = encoder.parameters()

    def get_discriminator_acc(self) -> torch.Tensor:
        acc = []
        for op in self._modules.values():
            if isinstance(op, paper.DiscriminatorEncodingOperator):
                acc.append(op.get_current_acc())
        return torch.mean(torch.stack(acc))

    def process_input_image(
        self, input_image: torch.Tensor, real: Optional[bool] = None
    ) -> pystiche.LossDict:
        return pystiche.LossDict(
            [(name, op(input_image, real)) for name, op in self.named_children()]
        )

    def forward(
        self, input_image: torch.Tensor, real: Optional[bool] = None
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        return self.process_input_image(input_image, real) * self.score_weight


def discriminator_operator(
    in_channels: int = 3,
    impl_params: bool = True,
    encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    prediction_modules: Union[Dict[str, nn.Module], None] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: Optional[float] = None,
) -> MultiLayerDicriminatorEncodingOperator:
    r"""Discriminator prediction from :cite:`SKL+2018`.

    Capture image details at different scales with the ``prediction_modules`` and sum up
    all losses and accuracies on the different ``layers``.

    Args:
        in_channels: Number of channels in the input.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        encoder: Trainable :class:`~pystiche.enc.MultiLayerEncoder`. If omitted, the
            default
            :class:`~pystiche_papers.sanakoyeu_et_al_2018.DiscriminatorMultiLayerEncoder`
            is used.
        layers: Layers from which the encodings of the ``encoder`` should be
            taken. If omitted, the defaults is used. Defaults to
            ``("0", "1", "3", "5", "6")``.
        prediction_modules: Auxiliary classifier used to process the encodings into a
            prediction. If omitted, the default
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.get_prediction_modules` are
            used.
        layer_weights: Layer weights of the operator. Defaults to ``sum``.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``1e0`` if
            ``impl_params is True`` otherwise ``1e-3``.
    """
    if encoder is None:
        encoder = paper.DiscriminatorMultiLayerEncoder(in_channels=in_channels)

    if score_weight is None:
        if impl_params:
            # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L98
            score_weight = 1e0
        else:
            score_weight = 1e-3

    if layers is None:
        layers = ("0", "1", "3", "5", "6")

    if prediction_modules is None:
        prediction_modules = paper.get_prediction_modules()

    assert tuple(prediction_modules.keys()) == layers, (
        "The keys in prediction_modules should match "
        "the entries in layers. However layers "
        + str(layers)
        + " and keys: "
        + str(tuple(prediction_modules.keys()))
        + " are given. "
    )

    def get_encoding_op(
        encoder: Encoder, layer_weight: float
    ) -> DiscriminatorEncodingOperator:
        prediction_module = prediction_modules[cast(str, encoder.layer)]  # type: ignore[index]
        return DiscriminatorEncodingOperator(
            encoder, prediction_module, score_weight=layer_weight,
        )

    return MultiLayerDicriminatorEncodingOperator(
        encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class DiscriminatorLoss(nn.Module):
    r"""Discriminator loss from :cite:`SKL+2018`.

    Calculates the loss and accuracy of the current discriminator on all real and fake
    input images.

    Args:
        discriminator: Trainable :class:`MultiLayerDicriminatorEncodingOperator`.
    """

    def __init__(self, discriminator: MultiLayerDicriminatorEncodingOperator) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.acc = torch.empty(1)

    @property
    def get_current_acc(self) -> torch.Tensor:
        return self.acc

    def calculate_loss(
        self,
        discriminator: MultiLayerDicriminatorEncodingOperator,
        output_photo: torch.Tensor,
        input_painting: torch.Tensor,
        input_photo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = discriminator(input_painting, real=True)
        acc = discriminator.get_discriminator_acc()
        for key, value in zip(
            loss.keys(), discriminator(output_photo, real=False).values()
        ):
            loss[key] = loss[key] + value

        acc += discriminator.get_discriminator_acc()
        if input_photo is not None:
            for key, value in zip(
                loss.keys(), discriminator(input_photo, real=False).values()
            ):
                loss[key] = loss[key] + value
            acc += discriminator.get_discriminator_acc()
            self.acc = acc / 3
            return cast(torch.Tensor, loss)
        self.acc = acc / 2
        return cast(torch.Tensor, loss)

    def forward(
        self,
        output_photo: torch.Tensor,
        input_painting: torch.Tensor,
        input_photo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.calculate_loss(
            self.discriminator, output_photo, input_painting, input_photo
        )
