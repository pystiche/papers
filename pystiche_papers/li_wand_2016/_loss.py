from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
from pystiche.enc import Encoder, MultiLayerEncoder
from pystiche.image.transforms import Transform
from pystiche.loss import PerceptualLoss
from pystiche.ops import FeatureReconstructionOperator as _FeatureReconstructionOperator
from pystiche.ops import MRFOperator as _MRFOperator
from pystiche.ops import MultiLayerEncodingOperator
from pystiche.ops import TotalVariationOperator as _TotalVariationOperator

from ._utils import extract_normalized_patches2d
from ._utils import multi_layer_encoder as multi_layer_encoder_

__all__ = [
    "FeatureReconstructionOperator",
    "content_loss",
    "MRFOperator",
    "style_loss",
    "TotalVariationOperator",
    "regularization",
    "perceptual_loss",
]


class FeatureReconstructionOperator(_FeatureReconstructionOperator):
    def __init__(
        self,
        encoder: Encoder,
        impl_params: bool = True,
        **feature_reconstruction_op_kwargs: Any,
    ):
        super().__init__(encoder, **feature_reconstruction_op_kwargs)

        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return mse_loss(input_repr, target_repr, reduction=self.loss_reduction)


def content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu4_2",
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    encoder = multi_layer_encoder.extract_encoder(layer)

    if score_weight is None:
        score_weight = 2e1 if impl_params else 1e0

    return FeatureReconstructionOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
    )


class MRFOperator(_MRFOperator):
    def __init__(
        self,
        encoder: Encoder,
        patch_size: Union[int, Tuple[int, int]],
        impl_params: bool = True,
        **mrf_op_kwargs: Any,
    ):

        super().__init__(encoder, patch_size, **mrf_op_kwargs)

        self.normalize_patches_grad = impl_params
        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def enc_to_repr(self, enc: torch.Tensor, is_guided: bool) -> torch.Tensor:
        if self.normalize_patches_grad:
            repr = extract_normalized_patches2d(enc, self.patch_size, self.stride)
        else:
            repr = pystiche.extract_patches2d(enc, self.patch_size, self.stride)
        if not is_guided:
            return repr

        return self._guide_repr(repr)

    def calculate_score(
        self,
        input_repr: torch.Tensor,
        target_repr: torch.Tensor,
        ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = F.mrf_loss(input_repr, target_repr, reduction=self.loss_reduction)
        return score * self.score_correction_factor


def style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    patch_size: Union[int, Tuple[int, int]] = 3,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    target_transforms: Optional[Iterable[Transform]] = None,
    score_weight: Optional[float] = None,
) -> MultiLayerEncodingOperator:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()

    if layers is None:
        layers = ("relu3_1", "relu4_1")

    if stride is None:
        stride = 2 if impl_params else 1

    if target_transforms is None:
        num_scale_steps = 1 if impl_params else 3
        scale_step_width = 5e-2
        num_rotate_steps = 1 if impl_params else 2
        rotate_step_width = 7.5
        target_transforms = MRFOperator.scale_and_rotate_transforms(
            num_scale_steps=num_scale_steps,
            scale_step_width=scale_step_width,
            num_rotate_steps=num_rotate_steps,
            rotate_step_width=rotate_step_width,
        )

    def get_encoding_op(encoder: Encoder, layer_weight: float) -> MRFOperator:
        return MRFOperator(
            encoder,
            patch_size,
            impl_params=impl_params,
            stride=stride,
            target_transforms=target_transforms,
            score_weight=layer_weight,
        )

    if score_weight is None:
        score_weight = 1e-4 if impl_params else 1e0

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class TotalVariationOperator(_TotalVariationOperator):
    def __init__(self, impl_params: bool = True, **total_variation_op_kwargs: Any):
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        score = F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor


def regularization(
    impl_params: bool = True, exponent: float = 2.0, score_weight: float = 1e-3,
) -> TotalVariationOperator:
    return TotalVariationOperator(
        impl_params=impl_params, exponent=exponent, score_weight=score_weight
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    regularization_kwargs: Optional[Dict[str, Any]] = None,
) -> PerceptualLoss:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()

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

    if regularization_kwargs is None:
        regularization_kwargs = {}
    regularization_ = regularization(impl_params=impl_params, **regularization_kwargs)

    return PerceptualLoss(content_loss_, style_loss_, regularization_)
