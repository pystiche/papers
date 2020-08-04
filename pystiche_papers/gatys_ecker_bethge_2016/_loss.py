import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

import torch
from torch import nn
from torch.nn.functional import mse_loss

import pystiche
from pystiche import enc, loss, ops

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

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0
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
    layer: str = "relu4_2",
    score_weight: float = 1e0,
) -> FeatureReconstructionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    encoder = multi_layer_encoder.extract_encoder(layer)

    return FeatureReconstructionOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
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

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        score = super().process_input_image(input_image)
        return score * self.score_correction_factor


def get_layer_weights(
    layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None
) -> List[float]:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    nums_channels = []
    for layer in layers:
        if layer not in multi_layer_encoder:
            raise RuntimeError

        layer = layer.replace("relu", "conv")
        if not layer.startswith("conv"):
            raise RuntimeError

        module = cast(nn.Conv2d, multi_layer_encoder._modules[layer])
        nums_channels.append(module.out_channels)

    return [1.0 / num_channels ** 2.0 for num_channels in nums_channels]


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_loss_kwargs: Any,
) -> StyleLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if layers is None:
        layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

    if layer_weights is None:
        try:
            layer_weights = get_layer_weights(
                layers, multi_layer_encoder=multi_layer_encoder
            )
        except RuntimeError:
            layer_weights = "mean"
            warnings.warn("ADDME", RuntimeWarning)

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> ops.GramOperator:
        return ops.GramOperator(encoder, score_weight=layer_weight, **gram_loss_kwargs)

    return StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder(impl_params=impl_params)

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss_ = content_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return loss.PerceptualLoss(content_loss_, style_loss_)
