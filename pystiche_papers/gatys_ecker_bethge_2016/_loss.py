from typing import Any, Callable, Optional, Sequence

import torch
from torch.nn.functional import mse_loss

import pystiche
from pystiche import enc, loss
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "FeatureReconstructionLoss",
    "content_loss",
    "MultiLayerEncodingLoss",
    "style_loss",
    "perceptual_loss",
]


class FeatureReconstructionLoss(loss.FeatureReconstructionLoss):
    r"""Feature reconstruction loss from :cite:`GEB2016`.

    Args:
        encoder: Encoder used to encode the input.
        impl_params: If ``False``, calculate the score with the squared error (SE)
            instead of the mean squared error (MSE). Furthermore, use a score
            correction factor of 1/2.
        **feature_reconstruction_loss_kwargs: Additional parameters of a
            :class:`pystiche.ops.FeatureReconstructionOperator`.

    .. seealso::

        - :class:`pystiche.loss.FeatureReconstructionLoss`
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        impl_params: bool = True,
        **feature_reconstruction_loss_kwargs: Any,
    ):
        super().__init__(encoder, **feature_reconstruction_loss_kwargs)

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
) -> FeatureReconstructionLoss:
    r"""Content loss from :cite:`GEB2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_ecker_bethge_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If
            omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.gatys_ecker_bethge_2016.FeatureReconstructionLoss`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return FeatureReconstructionLoss(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        impl_params=impl_params,
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class MultiLayerEncodingLoss(loss.MultiLayerEncodingLoss):
    r"""Multi-layer encoding loss from :cite:`GEB2016`.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        layers: Layers of the ``multi_layer_encoder`` that the children losses
            operate on.
        encoding_loss_fn: Callable that returns a children operator given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder`` and its corresponding layer weight.
        impl_params: If ``False``, use a score correction factor of 1/4.
        **multi_layer_encoding_op_kwargs: Additional parameters of a
            :class:`pystiche.ops.MultiLayerEncodingOperator`.

    .. seealso::

        - :class:`pystiche.loss.MultiLayerEncodingLoss`
    """

    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        layers: Sequence[str],
        encoding_loss_fn: Callable[[enc.Encoder, float], loss.Loss],
        impl_params: bool = True,
        **multi_layer_encoding_op_kwargs: Any,
    ) -> None:
        super().__init__(
            multi_layer_encoder,
            layers,
            encoding_loss_fn,
            **multi_layer_encoding_op_kwargs,
        )
        # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
        # Cell [3]
        # torch.nn.MSELoss() was used to calculate the style loss, which does not
        # include the factor 1/4 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        score = super().forward(input_image)
        return score * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> MultiLayerEncodingLoss:
    r"""Style loss from :cite:`GEB2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_ecker_bethge_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.gatys_ecker_bethge_2016.MultiLayerEncodingLoss`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def encoding_loss_fn(encoder: enc.Encoder, layer_weight: float) -> loss.GramLoss:
        return loss.GramLoss(encoder, score_weight=layer_weight)

    return MultiLayerEncodingLoss(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        encoding_loss_fn,
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
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_ecker_bethge_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    .. seealso::

        - :func:`pystiche_papers.gatys_ecker_bethge_2016.content_loss`
        - :func:`pystiche_papers.gatys_ecker_bethge_2016.style_loss`
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
