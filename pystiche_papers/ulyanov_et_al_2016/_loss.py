from typing import Any, Dict, Optional, Sequence, Union

import torch

from pystiche import enc, image, loss, ops

from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "FeatureReconstructionOperator",
    "content_loss",
    "GramOperator",
    "style_loss",
    "perceptual_loss",
]


class FeatureReconstructionOperator(ops.FeatureReconstructionOperator):
    def __init__(
        self, encoder: enc.Encoder, impl_params: bool = True, score_weight: float = 1e0
    ):
        super().__init__(encoder, score_weight=score_weight)
        self.double_batch_size_mean = impl_params

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = super().calculate_score(input_repr, target_repr, ctx)
        if not self.double_batch_size_mean:
            return score

        batch_size = image.extract_batch_size(input_repr)
        return score / batch_size


def content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layer: str = "relu4_2",
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
    if score_weight is None:
        if impl_params:
            score_weight = 1e0 if instance_norm else 6e-1
        else:
            score_weight = 1e0

    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    encoder = multi_layer_encoder.extract_encoder(layer)

    return FeatureReconstructionOperator(
        encoder, score_weight=score_weight, impl_params=impl_params
    )


class GramOperator(ops.GramOperator):
    def __init__(
        self, encoder: enc.Encoder, impl_params: bool = True, **gram_op_kwargs: Any,
    ):
        super().__init__(encoder, **gram_op_kwargs)
        self.normalize_by_num_channels = impl_params
        self.loss_reduction = "mean"
        self.double_batch_size_mean = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        gram_matrix = super().enc_to_repr(enc)
        if not self.normalize_by_num_channels:
            return gram_matrix

        num_channels = gram_matrix.size()[-1]
        return gram_matrix / num_channels

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = super().calculate_score(input_repr, target_repr, ctx)
        if not self.double_batch_size_mean:
            return score

        batch_size = input_repr.size()[0]
        return score / batch_size


def style_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: Optional[float] = None,
    **gram_op_kwargs: Any,
) -> ops.MultiLayerEncodingOperator:
    if score_weight is None:
        if impl_params:
            if instance_norm:
                score_weight = 1e0
            else:
                score_weight = 1e3 if stylization else 1e0
        else:
            score_weight = 1e0

    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if layers is None:
        if impl_params and instance_norm:
            layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1")
        else:
            layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> GramOperator:
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    stylization: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        stylization=stylization,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if stylization:
        if content_loss_kwargs is None:
            content_loss_kwargs = {}
        content_loss_ = content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            multi_layer_encoder=multi_layer_encoder,
            **content_loss_kwargs,
        )
    else:
        content_loss_ = None  # type: ignore[assignment]

    return loss.PerceptualLoss(content_loss_, style_loss_)
