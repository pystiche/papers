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
    r"""Feature reconstruction operator from :cite:`LW2016`.

    Args:
        encoder: Encoder used to encode the input.
        impl_params: If ``False``, calculate the score with the squared error (SE)
            instead of the mean squared error (MSE).
        **feature_reconstruction_op_kwargs: Additional parameters of a
            :class:`pystiche.ops.FeatureReconstructionOperator`.

    .. seealso::

        :class:`pystiche.ops.FeatureReconstructionOperator`
    """

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
    r"""Content loss from :cite:`LW2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If
            omitted, :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    .. seealso::

        :class:`pystiche_papers.li_wand_2016.FeatureReconstructionOperator`
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
    r"""MRF operator from :cite:`LW2016`.

    Args:
        encoder: Encoder used to encode the input.
        patch_size: Spatial size of the neural patches.
        impl_params: If ``True``, normalize the gradient of the neural patches. If
            ``False``, use a score correction factor of 1/2.
        **mrf_op_kwargs: Additional parameters of a :class:`pystiche.ops.MRFOperator`.

    In contrast to :class:`pystiche.ops.MRFOperator`, the the score is calculated
    with the squared error (SE) instead of the mean squared error (MSE).

    .. seealso::

        - :class:`pystiche.ops.MRFOperator`
        - :func:`pystiche_papers.li_wand_2016.extract_normalized_patches2d`
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        patch_size: Union[int, Tuple[int, int]],
        impl_params: bool = True,
        **mrf_op_kwargs: Any,
    ):
        super().__init__(encoder, patch_size, **mrf_op_kwargs)

        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/mrf.lua#L221
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/mrf.lua#L224
        # They use normalized patches instead of the unnormalized patches described in
        # the paper.
        self.normalize_patches_grad = impl_params
        self.loss_reduction = "sum"

        # The score correction factor is not visible in the reference implementation
        # of the original authors, since the calculation is performed with respect to
        # the gradient and not the score. Roughly speaking, since the calculation
        # comprises a *squared* distance, we need a factor of 1/2 in the forward pass.
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/mrf.lua#L220
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
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    r"""Style loss from :cite:`LW2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If
            omitted, :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.li_wand_2016.MRFOperator`
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
    r"""Total variation operator from :cite:`LW2016`.

    Args:
        impl_params: If ``False``, use a score correction factor of 1/2.
        **total_variation_op_kwargs: Additional parameters of a
            :class:`pystiche.ops.TotalVariationOperator`.

    In contrast to :class:`pystiche.ops.TotalVariationOperator`, the the score is
    calculated with the squared error (SE) instead of the mean squared error (MSE).

    .. seealso::

        - :class:`pystiche.ops.TotalVariationOperator`
    """

    def __init__(self, impl_params: bool = True, **total_variation_op_kwargs: Any):
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"

        # The score correction factor is not visible in the reference implementation
        # of the original authors, since the calculation is performed with respect to
        # the gradient and not the score. Roughly speaking, since the calculation
        # comprises a *squared* distance, we need a factor of 1/2 in the forward pass.
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/tv.lua#L20-L30
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

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
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.li_wand_2016.TotalVariationOperator`
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
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        multi_layer_encoder: Pretrained multi-layer encoder. If
            omitted, :func:`~pystiche_papers.li_wand_2016.multi_layer_encoder` is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    .. seealso::

        - :func:`pystiche_papers.li_wand_2016.content_loss`
        - :func:`pystiche_papers.li_wand_2016.style_loss`
        - :func:`pystiche_papers.li_wand_2016.regularization`
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
