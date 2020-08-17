from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
from pystiche import enc, loss, ops
from pystiche.image import transforms

from ._utils import extract_normalized_patches2d
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "FeatureReconstructionOperator",
    "content_loss",
    "MRFOperator",
    "style_loss",
    "TotalVariationOperator",
    "regularization",
    "perceptual_loss",
]


class FeatureReconstructionOperator(ops.FeatureReconstructionOperator):
    def __init__(
        self,
        encoder: enc.Encoder,
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
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layer: str = "relu4_2",
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
    r"""Content_loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        layer: Layer from which the encodings of the ``multi_layer_encoder`` should be
            taken. Defaults to ``"relu4_2"``.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to ``impl_params``. Defaults to ``2e1`` if
            ``impl_params is True`` otherwise ``1e0``.

        If ``impl_params is True`` , a loss reduction of ``"mean"`` is used instead of
        ``"sum"``.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()
    encoder = multi_layer_encoder.extract_encoder(layer)

    if score_weight is None:
        score_weight = 2e1 if impl_params else 1e0

    return FeatureReconstructionOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
    )


class MRFOperator(ops.MRFOperator):
    def __init__(
        self,
        encoder: enc.Encoder,
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
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    patch_size: Union[int, Tuple[int, int]] = 3,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    target_transforms: Optional[Iterable[transforms.Transform]] = None,
    score_weight: Optional[float] = None,
) -> ops.MultiLayerEncodingOperator:
    r"""Style_loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted,  the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        layers: Layers from which the encodings of the ``multi_layer_encoder`` should be
            taken. If omitted, the defaults is used. Defaults to
            ``("relu3_1", "relu4_1")``.
        layer_weights: Layer weights of the operator. Defaults to ``"sum"``.
        patch_size: Size of a patch. Defaults to ``3``.
        stride: Stride of the convolution. If omitted, the stride is determined with
            respect to `impl_params``. Defaults to ``2`` if ``impl_params is True``
            otherwise ``1``.
        target_transforms: Optional augemntation transformations for the target. If
            omitted, the transforms are determined with respect to ``impl_params``. For
            details see below.
        score_weight: Score weight of the operator. If omitted, the score_weight is
            determined with respect to `impl_params``. Defaults to ``1e-4`` if
            ``impl_params is True`` otherwise ``1e0``.

    If ``impl_params is True`` , an additional score correction factor of ``1.0 / 2.0``
    is used instead. Additionally, no transformations are used in the scaling
    and rotation. In the paper two additional transformations are used in the scaling
    and the rotation. Furthermore, normalized patches are used instead of the
    unnormalized  patches. For details see
    `extract_normalized_patches2d <https://github.com/pmeier/pystiche_papers/blob/b15202068d683c8b60f8a437e562f3d922fe0f3e/pystiche_papers/li_wand_2016/_utils.py#L51>`_
    .

    The parameters ``patch_size`` and ``stride`` can either be:

    * a single :class:`int` – in which case the same value is used for the height and width
      dimension
    * a tuple of two ints – in which case, the first int is used for the vertical
      dimension, and the second int for the horizontal dimension
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

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

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> MRFOperator:
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

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


class TotalVariationOperator(ops.TotalVariationOperator):
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
    r"""Regularization from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        exponent: Parameter :math:`\beta` . A higher value leads to more smoothed
            results. Defaults to ``2.0``.
        score_weight: Score weight of the operator. Defaults to ``1e-3``.

    If ``impl_params is True`` , an additional score correction factor of ``1.0 / 2.0``
    is used.
    """
    return TotalVariationOperator(
        impl_params=impl_params, exponent=exponent, score_weight=score_weight
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
    regularization_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        content_loss_kwargs: Optional parameters for the :func:`content_loss`.
        style_loss_kwargs: Optional parameters for the :func:`style_loss`.
        regularization_kwargs: Optional parameters for the :func:`regularization`.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

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

    return loss.PerceptualLoss(content_loss_, style_loss_, regularization_)
