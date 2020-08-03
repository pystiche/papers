from typing import Any, Callable, Dict, Optional, Sequence, Union, cast

import torch

import pystiche
from pystiche import enc, loss, ops

from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "StyleLoss",
    "style_loss",
    "guided_style_loss",
    "perceptual_loss",
    "guided_perceptual_loss",
]


def content_loss(
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layer: str = "relu4_2",
    score_weight: float = 1e0,
) -> ops.FeatureReconstructionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    encoder = multi_layer_encoder.extract_encoder(layer)

    return ops.FeatureReconstructionOperator(encoder, score_weight=score_weight)


class StyleLoss(ops.MultiLayerEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: enc.MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[enc.Encoder, float], ops.EncodingOperator],
        impl_params: bool = True,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e0,
    ) -> None:
        if layer_weights is None:
            layer_weights = self.get_default_layer_weights(multi_layer_encoder, layers)

        super().__init__(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    @staticmethod
    def get_default_layer_weights(
        multi_layer_encoder: enc.MultiLayerEncoder, layers: Sequence[str]
    ) -> Sequence[float]:
        nums_channels = []
        for layer in layers:
            module = multi_layer_encoder._modules[layer.replace("relu", "conv")]
            nums_channels.append(cast(int, module.out_channels))
        return [1.0 / num_channels ** 2.0 for num_channels in nums_channels]

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return super().process_input_image(input_image) * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
) -> StyleLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if layers is None:
        layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def guided_style_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    region_weights: Union[str, Sequence[float]] = "sum",
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
) -> ops.MultiRegionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    def get_region_op(region: str, region_weight: float) -> StyleLoss:
        return style_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layers=layers,
            layer_weights=layer_weights,
            score_weight=region_weight,
            **gram_op_kwargs,
        )

    return ops.MultiRegionOperator(
        regions, get_region_op, region_weights=region_weights, score_weight=score_weight
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss_ = content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return loss.PerceptualLoss(content_loss_, style_loss_)


def guided_perceptual_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.GuidedPerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss_ = content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = guided_style_loss(
        regions,
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return loss.GuidedPerceptualLoss(content_loss_, style_loss_)
