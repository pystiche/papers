from typing import Any, Callable, Optional, Sequence, Union

import torch
from torch.nn.functional import mse_loss

import pystiche
from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "FeatureReconstructionOperator",
    "content_loss",
    "StyleLoss",
    "style_loss",
    "perceptual_loss",
]


class FeatureReconstructionOperator(ops.FeatureReconstructionOperator):
    def __init__(
        self, encoder: enc.Encoder, impl_params: bool = True, score_weight: float = 1e0
    ):
        super().__init__(encoder, score_weight=score_weight)

        # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
        # Cell [8]
        # torch.nn.MSELoss() was used to calculate the content loss, which does not
        # include the factor 1/2 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0
        # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
        # Cell [8]
        # torch.nn.MSELoss() was used to calculate the content loss, which by default
        # uses reduction="mean"
        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = mse_loss(input_repr, target_repr, reduction=self.loss_reduction)
        return score * self.score_correction_factor


def content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> FeatureReconstructionOperator:
    r"""Content_loss from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        layer: Layer from which the encodings of the ``multi_layer_encoder`` should be
            taken. Defaults to ``"relu4_2"``.
        score_weight: Score weight of the operator. Defaults to ``1e0``.

    If ``impl_params is True``,

    * no additional score correction factor of ``1.0 / 2.0`` is used, and
    * a loss reduction ``"mean"`` is used instead of a ``"sum"``.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        impl_params=impl_params,
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class StyleLoss(ops.MultiLayerEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[enc.Encoder, float], ops.EncodingOperator],
        impl_params: bool = True,
        layer_weights: Union[str, Sequence[float]] = "mean",
        score_weight: float = 1e0,
    ) -> None:
        super().__init__(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )
        # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
        # Cell [3]
        # torch.nn.MSELoss() was used to calculate the style loss, which does not
        # include the factor 1/4 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        score = super().process_input_image(input_image)
        return score * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
    **gram_loss_kwargs: Any,
) -> StyleLoss:
    r"""Style_loss from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        layers: Layers from which the encodings of the ``multi_layer_encoder`` should be
            taken. If omitted, the defaults is used. Defaults to
            ``("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")``.
        layer_weights: Layer weights of the operator. If omitted, the layer weights are
            calculated with ``1.0 / num_channels ** 2.0`` as described in the paper.
            Here the ``num_channels`` are the number of channels in the respective
            layer.
        score_weight: Score weight of the operator. Defaults to ``1e3``.
        **gram_loss_kwargs: Optional parameters for the
            :class:`~pystiche.ops.GramOperator`.

    If ``impl_params is True`` , no additional score correction factor of ``1.0 / 4.0``
    is used.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight, **gram_loss_kwargs)

    return StyleLoss(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=hyper_parameters.style_loss.layer_weights,
        score_weight=hyper_parameters.style_loss.score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        content_loss_kwargs: Optional parameters for the :func:`content_loss`.
        style_loss_kwargs: Optional parameters for the :func:`style_loss`.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return loss.PerceptualLoss(
        content_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
        style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
    )
