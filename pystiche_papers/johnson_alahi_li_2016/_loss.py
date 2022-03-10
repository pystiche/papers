from typing import Any, Optional

import torch

import pystiche.loss.functional as F
from pystiche import enc, loss
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "GramLoss",
    "style_loss",
    "TotalVariationLoss",
    "regularization",
    "perceptual_loss",
]


def content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.FeatureReconstructionLoss:
    r"""Content loss from :cite:`JAL2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <johnson_alahi_li_2016-impl_params>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder` is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return loss.FeatureReconstructionLoss(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class GramLoss(loss.GramLoss):
    r"""Gram operator from :cite:`JAL2016`.

    Args:
        encoder: Encoder used to encode the input.
        impl_params: If ``True``, normalize the Gram matrix additionally by the number
            of channels.
        **gram_op_kwargs: Additional parameters of a :class:`pystiche.loss.GramLoss`.

    .. seealso::

        - :class:`pystiche.loss.GramOperator`
    """

    def __init__(
        self, encoder: enc.Encoder, impl_params: bool = True, **gram_op_kwargs: Any,
    ) -> None:
        super().__init__(encoder, **gram_op_kwargs)
        self.normalize_by_num_channels = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        gram_matrix = super().enc_to_repr(enc)
        if not self.normalize_by_num_channels:
            return gram_matrix

        num_channels = gram_matrix.size()[-1]
        return gram_matrix / num_channels


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.MultiLayerEncodingLoss:
    r"""Style loss from :cite:`JAL2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <johnson_alahi_li_2016-impl_params>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder` is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> GramLoss:
        return GramLoss(encoder, impl_params=impl_params, score_weight=layer_weight)

    return loss.MultiLayerEncodingLoss(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.style_loss.layer_weights,
        score_weight=hyper_parameters.style_loss.score_weight,
    )


class TotalVariationLoss(loss.TotalVariationLoss):
    r"""Total variation operator from :cite:`LW2016`.

    Args:
        **total_variation_op_kwargs: Additional parameters of a
            :class:`pystiche.loss.TotalVariationOperator`.

    In contrast to :class:`pystiche.loss.TotalVariationOperator`, the the score is
    calculated with the squared error (SE) instead of the mean squared error (MSE).

    .. seealso::

        - :class:`pystiche.loss.TotalVariationOperator`
    """

    def __init__(self, **total_variation_op_kwargs: Any) -> None:
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )


def regularization(
    hyper_parameters: Optional[HyperParameters] = None,
) -> TotalVariationLoss:
    r"""Regularization from :cite:`JAL2016`.

    Args:
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return TotalVariationLoss(score_weight=hyper_parameters.regularization.score_weight)


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss comprising content and style loss as well as a regularization.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <johnson_alahi_li_2016-impl_params>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016._utils.multi_layer_encoder`
            is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
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
        regularization(hyper_parameters=hyper_parameters,),
    )
