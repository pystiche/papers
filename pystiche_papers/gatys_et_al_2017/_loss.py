from copy import copy
from typing import Any, Callable, Optional, Sequence

import torch

import pystiche
from pystiche import enc, loss
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "MultiLayerEncodingLoss",
    "style_loss",
    "guided_style_loss",
    "perceptual_loss",
    "guided_perceptual_loss",
]


def content_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.FeatureReconstructionLoss:
    r"""Content_loss from :cite:`GEB+2017`.

    Args:
        multi_layer_encoder: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return loss.FeatureReconstructionLoss(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class MultiLayerEncodingLoss(loss.MultiLayerEncodingLoss):
    r"""Multi-layer encoding Loss from :cite:`GEB2016`.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        layers: Layers of the ``multi_layer_encoder`` that the children losses
            operate on.
        get_encoding_op: Callable that returns a children Loss given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder`` and its corresponding layer weight.
        impl_params: If ``False``, use a score correction factor of 1/4.
        **multi_layer_encoding_op_kwargs: Additional parameters of a
            :class:`pystiche.loss.MultiLayerEncodingLoss`.

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
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/LossLayers.lua#L63
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/LossLayers.lua#L75
        # torch.nn.MSELoss() was used as criterion for the content loss, which does not
        # include the factor 1/4 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def forward(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return super().forward(input_image) * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> MultiLayerEncodingLoss:
    r"""Style_loss from :cite:`GEB+2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        multi_layer_encoder: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.gatys_et_al_2017.MultiLayerEncodingLoss`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> loss.GramLoss:
        return loss.GramLoss(encoder, score_weight=layer_weight)

    return MultiLayerEncodingLoss(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=hyper_parameters.style_loss.layer_weights,
        score_weight=hyper_parameters.style_loss.score_weight,
    )


def guided_style_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.MultiRegionLoss:
    r"""Guided style_loss from :cite:`GEB+2017`.

    Args:
        regions: Regions of the input image to be stylized.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        multi_layer_encoder: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.gatys_et_al_2017.MultiLayerEncodingLoss`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    # the copy is needed here in order to not override style_loss.score_weight later
    hyper_parameters = (
        _hyper_parameters() if hyper_parameters is None else copy(hyper_parameters)
    )

    def get_region_op(region: str, region_weight: float) -> MultiLayerEncodingLoss:
        hyper_parameters.style_loss.score_weight = region_weight  # type: ignore[union-attr]
        return style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters.new_similar(),  # type: ignore[union-attr]
        )

    return loss.MultiRegionLoss(
        regions,
        get_region_op,
        region_weights=hyper_parameters.guided_style_loss.region_weights,
        score_weight=hyper_parameters.guided_style_loss.score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`GEB+2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        multi_layer_encoder: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return loss.PerceptualLoss(
        content_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
        style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
    )


def guided_perceptual_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Guided perceptual loss from :cite:`GEB+2017`.

    Args:
        regions: Regions of the input image to be stylized.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <gatys_et_al_2017-impl_params>`.
        multi_layer_encoder: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.multi_layer_encoder` is
            used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_ecker_bethge_2016.hyper_parameters` is used.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return loss.PerceptualLoss(
        content_loss(
            multi_layer_encoder=multi_layer_encoder, hyper_parameters=hyper_parameters,
        ),
        guided_style_loss(
            regions,
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
    )
