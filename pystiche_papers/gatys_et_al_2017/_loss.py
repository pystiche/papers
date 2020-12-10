from copy import copy
from typing import Any, Callable, Optional, Sequence

import torch

import pystiche
from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "MultiLayerEncodingOperator",
    "style_loss",
    "guided_style_loss",
    "perceptual_loss",
    "guided_perceptual_loss",
]


def content_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.FeatureReconstructionOperator:
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

    return ops.FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class MultiLayerEncodingOperator(ops.MultiLayerEncodingOperator):
    r"""Multi-layer encoding operator from :cite:`GEB2016`.

    Args:
        multi_layer_encoder: Multi-layer encoder.
        layers: Layers of the ``multi_layer_encoder`` that the children operators
            operate on.
        get_encoding_op: Callable that returns a children operator given a
            :class:`pystiche.enc.SingleLayerEncoder` extracted from the
            ``multi_layer_encoder`` and its corresponding layer weight.
        impl_params: If ``False``, use a score correction factor of 1/4.
        **multi_layer_encoding_op_kwargs: Additional parameters of a
            :class:`pystiche.ops.MultiLayerEncodingOperator`.

    .. seealso::

        - :class:`pystiche.ops.MultiLayerEncodingOperator`
    """

    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[enc.Encoder, float], ops.EncodingOperator],
        impl_params: bool = True,
        **multi_layer_encoding_op_kwargs: Any,
    ) -> None:
        super().__init__(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            **multi_layer_encoding_op_kwargs,
        )
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/LossLayers.lua#L63
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/LossLayers.lua#L75
        # torch.nn.MSELoss() was used as criterion for the content loss, which does not
        # include the factor 1/4 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return super().process_input_image(input_image) * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> MultiLayerEncodingOperator:
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

        - :class:`pystiche_papers.gatys_et_al_2017.MultiLayerEncodingOperator`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight)

    return MultiLayerEncodingOperator(
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
) -> ops.MultiRegionOperator:
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

        - :class:`pystiche_papers.gatys_et_al_2017.MultiLayerEncodingOperator`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    # the copy is needed here in order to not override style_loss.score_weight later
    hyper_parameters = (
        _hyper_parameters() if hyper_parameters is None else copy(hyper_parameters)
    )

    def get_region_op(region: str, region_weight: float) -> MultiLayerEncodingOperator:
        hyper_parameters.style_loss.score_weight = region_weight  # type: ignore[union-attr]
        return style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters.new_similar(),  # type: ignore[union-attr]
        )

    return ops.MultiRegionOperator(
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
) -> loss.GuidedPerceptualLoss:
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

    return loss.GuidedPerceptualLoss(
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
