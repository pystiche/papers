from typing import Any, Optional, Tuple

import torch

from pystiche import enc, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import _hyper_parameters, multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "content_loss",
    "GramOperator",
    "style_loss",
    "perceptual_loss",
]


# https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L57
class ManipulateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, score_weight: float) -> torch.Tensor:  # type: ignore[override]
        ctx.score_weight = score_weight
        return input_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        grad_output = grad_output / ctx.score_weight
        grad_input = grad_output / (torch.norm(grad_output, p=1) + 1e-8)
        grad_input = grad_input * ctx.score_weight
        return grad_input, None


manipulate_gradient = ManipulateGradient.apply


def content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.FeatureReconstructionOperator:
    r"""Content loss from :cite:`ULVL2016,UVL2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, :func:`~pystiche_papers.ulyanov_et_al_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.ulyanov_et_al_2016.FeatureReconstructionOperator`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    return ops.FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        score_weight=hyper_parameters.content_loss.score_weight,
    )


class GramOperator(ops.GramOperator):
    r"""Gram operator from :cite:`ULVL2016,UVL2017`.

    Args:
        encoder: Encoder used to encode the input.
        impl_params: If ``True``, normalize the score twice by the batch size.
        **gram_op_kwargs: Additional parameters of a :class:`pystiche.ops.GramOperator`.

    .. seealso::

        - :class:`pystiche.ops.GramOperator`
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        impl_params: bool = True,
        **gram_op_kwargs: Any,
    ):
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L38
        # In the reference implementation the gram_matrix is only divided by the
        # batch_size.
        self.normalize_by_num_channels = impl_params
        super().__init__(encoder, **gram_op_kwargs)

        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L56-L57
        # In the reference implementation the gradient of the style loss is
        # normalized.
        self.manipulate_gradient = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        if self.manipulate_gradient:
            enc = manipulate_gradient(enc, self.score_weight)
        gram_matrix = super().enc_to_repr(enc)
        if not self.normalize_by_num_channels:
            return gram_matrix

        num_channels = gram_matrix.size()[-1]
        return gram_matrix / num_channels


def style_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> ops.MultiLayerEncodingOperator:
    r"""Style loss from :cite:`ULVL2016,UVL2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, :func:`~pystiche_papers.ulyanov_et_al_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.

    .. seealso::

        - :class:`pystiche_papers.ulyanov_et_al_2016.GramOperator`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    def get_encoding_op(encoder: enc.Encoder, layer_weight: float) -> GramOperator:
        return GramOperator(encoder, impl_params=impl_params, score_weight=layer_weight)

    return ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        hyper_parameters.style_loss.layers,
        get_encoding_op,
        layer_weights=hyper_parameters.style_loss.layer_weights,
        score_weight=hyper_parameters.style_loss.score_weight,
    )


def perceptual_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`ULVL2016,UVL2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, :func:`~pystiche_papers.ulyanov_et_al_2016.multi_layer_encoder`
            is used.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.

    .. seealso::

        - :func:`pystiche_papers.ulyanov_et_al_2016.content_loss`
        - :func:`pystiche_papers.ulyanov_et_al_2016.style_loss`
    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    return loss.PerceptualLoss(
        content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
        style_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            multi_layer_encoder=multi_layer_encoder,
            hyper_parameters=hyper_parameters,
        ),
    )
