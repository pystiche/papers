from typing import Any, Dict, Optional, Sequence, Union

import torch

import pystiche.ops.functional as F
from pystiche import enc, loss, ops

from ._utils import _maybe_get_luatorch_param
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "GramOperator",
    "style_loss",
    "TotalVariationOperator",
    "regularization",
    "perceptual_loss",
]


LUATORCH_CONTENT_SCORE_WEIGHTS = {
    ("candy", True): 1.0,
    ("composition_vii", False): 1.0,
    ("feathers", True): 1.0,
    ("la_muse", False): 1.0,
    ("la_muse", True): 0.5,
    ("mosaic", True): 1.0,
    ("starry_night", False): 1.0,
    ("the_scream", True): 1.0,
    ("the_wave", False): 1.0,
    ("udnie", True): 0.5,
}


def get_content_score_weight(
    impl_params: bool,
    instance_norm: bool,
    style: Optional[str] = None,
    default: float = 1e0,
) -> float:
    # The paper reports no style score weight so we go with the default value of the
    # implementation instead
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L36
    return _maybe_get_luatorch_param(
        LUATORCH_CONTENT_SCORE_WEIGHTS, impl_params, instance_norm, style, default
    )


def content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layer: str = "relu2_2",
    score_weight: Optional[float] = None,
) -> ops.FeatureReconstructionOperator:
    r"""Content_loss from :cite:`JAL2016`.

    Args:
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper.
        style: Optional style for selecting the optimization parameters for the replication.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If ``omitted``, the
            default :func:`~pystiche_papers.johnson_alahi_li_2016.multi_layer_encoder` from the paper is used.
        layer: Layer from which the encodings of the ``multi_layer_encoder`` should be taken. Defaults to "relu2_2".
        score_weight: Score weight of the operator. If ``omitted``, the score_weight is determined with respect to
            ``style`` and ``instance_norm``. For details see FIXME

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)
    encoder = multi_layer_encoder.extract_encoder(layer)

    if score_weight is None:
        score_weight = get_content_score_weight(impl_params, instance_norm, style)

    return ops.FeatureReconstructionOperator(encoder, score_weight=score_weight)


class GramOperator(ops.GramOperator):
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


LUATORCH_STYLE_SCORE_WEIGHTS = {
    ("candy", True): 10.0,
    ("composition_vii", False): 5.0,
    ("feathers", True): 10.0,
    ("la_muse", False): 5.0,
    ("la_muse", True): 10.0,
    ("mosaic", True): 10.0,
    ("starry_night", False): 3.0,
    ("the_scream", True): 20.0,
    ("the_wave", False): 5.0,
    ("udnie", True): 10.0,
}


def get_style_score_weight(
    impl_params: bool,
    instance_norm: bool,
    style: Optional[str] = None,
    default: float = 5.0,
) -> float:
    # The paper reports no style score weight so we go with the default value of the
    # implementation instead
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L43
    return _maybe_get_luatorch_param(
        LUATORCH_STYLE_SCORE_WEIGHTS, impl_params, instance_norm, style, default
    )


def style_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: Optional[float] = None,
    **gram_op_kwargs: Any,
) -> ops.MultiLayerEncodingOperator:
    r"""Style_loss from :cite:`JAL2016`.

    Args:
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper.
        style: Optional style for selecting the optimization parameters for the replication.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If ``omitted``,
            the default :func:`~pystiche_papers.johnson_alahi_li_2016._utils.multi_layer_encoder` from the paper is used.
        layers: Layers from which the encodings of the ``multi_layer_encoder`` should be taken. If ``None``, the
            defaults is used. Defaults to ``("relu1_2", "relu2_2", "relu3_3", "relu4_3")``.
        layer_weights: Layer weights of the operator. Defaults to "sum".
        score_weight: Score weight of the operator. If ``omitted``, the score_weight is determined with respect to
            ``style``, ``instance_norm`` and ``impl_params``. For details see FIXME
        **gram_op_kwargs: Optional parameters for the :class:`~pystiche.ops.GramOperator`.
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if layers is None:
        layers = ("relu1_2", "relu2_2", "relu3_3", "relu4_3")

    if score_weight is None:
        score_weight = get_style_score_weight(impl_params, instance_norm, style)

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> GramOperator:
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class TotalVariationOperator(ops.TotalVariationOperator):
    def __init__(self, **total_variation_op_kwargs: Any) -> None:
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        return F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )


LUATORCH_REGULARIZATION_SCORE_WEIGHTS = {
    ("candy", True): 1e-4,
    ("composition_vii", False): 1e-6,
    ("feathers", True): 1e-5,
    ("la_muse", False): 1e-5,
    ("la_muse", True): 1e-4,
    ("mosaic", True): 1e-5,
    ("starry_night", False): 1e-5,
    ("the_scream", True): 1e-5,
    ("the_wave", False): 1e-4,
    ("udnie", True): 1e-6,
}


def get_regularization_score_weight(
    impl_params: bool,
    instance_norm: bool,
    style: Optional[str] = None,
    default: float = 1e-6,
) -> float:
    # The paper reports a range of regularization score weights so we go with the
    # default value of the implementation instead
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L33
    return _maybe_get_luatorch_param(
        LUATORCH_REGULARIZATION_SCORE_WEIGHTS,
        impl_params,
        instance_norm,
        style,
        default,
    )


def regularization(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    score_weight: Optional[float] = None,
    **total_variation_op_kwargs: Any,
) -> TotalVariationOperator:
    r"""Regularization from :cite:`JAL2016`.

    Args:
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper.
        style: Optional style for selecting the optimization parameters for the replication.
        score_weight: Score weight of the operator. If omitted, the score_weight is determined with respect to
            ``style`` and ``instance_norm``. For details see FIXME
        **total_variation_op_kwargs: Optional parameters for the ``ops.TotalVariationOperator``.

    """
    if score_weight is None:
        score_weight = get_regularization_score_weight(
            impl_params, instance_norm, style
        )
    return TotalVariationOperator(
        score_weight=score_weight, **total_variation_op_kwargs
    )


def perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    style: Optional[str] = None,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    total_variation_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss comprising content and style loss as well as a regularization.

    Args:
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper.
        style: Optional style for selecting the optimization parameters for the replication.
            of the pretrained transformers.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`.
        content_loss_kwargs: Optional parameters for the :func:`content_loss`.
        style_loss_kwargs: Optional parameters for the ``style_loss``.
        total_variation_kwargs: Optional parameters for the ``regularization``.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss_ = content_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if total_variation_kwargs is None:
        total_variation_kwargs = {}
    regularization_ = regularization(
        impl_params=impl_params,
        instance_norm=instance_norm,
        style=style,
        **total_variation_kwargs,
    )

    return loss.PerceptualLoss(content_loss_, style_loss_, regularization_)
