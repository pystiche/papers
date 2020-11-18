from typing import Any, Optional, Tuple, Union

import torch
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import extract_normalized_patches2d
from ._utils import hyper_parameters as _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder
from ._utils import target_transforms as _target_transforms

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

        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/content.lua#L15
        # nn.MSECriterion() was used as criterion to calculate the content loss, which
        # by default uses reduction="mean"
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
    hyper_parameters: Optional[HyperParameters] = None,
) -> FeatureReconstructionOperator:
    r"""Content_loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    If ``impl_params is True`` , a loss reduction of ``"mean"`` is used instead of
    ``"sum"``.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    return FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        impl_params=impl_params,
        score_weight=hyper_parameters.content_loss.score_weight,
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

        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/mrf.lua#L108
        # They use normalized patches instead of the unnormalized patches described in
        # the paper.
        self.normalize_patches_grad = impl_params
        self.loss_reduction = "sum"
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/style.lua#L34
        # nn.MSECriterion() was used as criterion to calculate the style loss, which
        # does not include the factor 1/2 given in the paper
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0

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
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    r"""Style_loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted,  the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    If ``impl_params is True``,

    * an additional score correction factor of ``1.0 / 2.0`` is used,
    * normalized patches are used (see
      :func:`~pystiche_papers.li_wand_2016._utils.extract_normalized_patches2d` for
      details), and
    * no target augmentation transformations are used.
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> MRFOperator:
        return MRFOperator(
            encoder,
            hyper_parameters.style_loss.patch_size,  # type: ignore[union-attr]
            impl_params=impl_params,
            stride=hyper_parameters.style_loss.stride,  # type: ignore[union-attr]
            target_transforms=_target_transforms(
                impl_params=impl_params, hyper_parameters=hyper_parameters
            ),
            score_weight=layer_weight,
        )

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.style_loss.layer_weights,
        score_weight=hyper_parameters.style_loss.score_weight,
    )


class TotalVariationOperator(ops.TotalVariationOperator):
    def __init__(self, impl_params: bool = True, **total_variation_op_kwargs: Any):
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0

    def calculate_score(self, input_repr: torch.Tensor) -> torch.Tensor:
        score = F.total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor


def regularization(
    impl_params: bool = True, hyper_parameters: Optional[HyperParameters] = None,
) -> TotalVariationOperator:
    r"""Regularization from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    If ``impl_params is True`` , an additional score correction factor of ``1.0 / 2.0``
    is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return TotalVariationOperator(
        impl_params=impl_params,
        score_weight=hyper_parameters.regularization.score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

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
        regularization(impl_params=impl_params, hyper_parameters=hyper_parameters),
    )
