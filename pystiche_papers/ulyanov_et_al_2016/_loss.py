from typing import Any, Optional, Tuple, Union, cast

import torch

import pystiche
from pystiche import enc, image, loss, ops
from pystiche_papers.utils import HyperParameters

from ._utils import _hyper_parameters
from ._utils import multi_layer_encoder as _multi_layer_encoder

__all__ = [
    "FeatureReconstructionOperator",
    "content_loss",
    "GramOperator",
    "style_loss",
    "perceptual_loss",
]


# https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L48-L55
class RemoveScoreWeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(self: Any, input_tensor: torch.Tensor, weight: float) -> torch.Tensor:  # type: ignore[override]
        self.weight = weight
        return input_tensor

    @staticmethod
    def backward(self: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:  # type: ignore[override]
        grad_input = grad_output.clone()
        return grad_input / self.weight, None


class AddScoreWeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(self: Any, input_tensor: torch.Tensor, weight: float) -> torch.Tensor:  # type: ignore[override]
        self.weight = weight
        return input_tensor

    @staticmethod
    def backward(self: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:  # type: ignore[override]
        grad_input = grad_output.clone()
        return grad_input * self.weight, None


# https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L48-L55
class RemoveBatchSizeDivisionGradient(torch.autograd.Function):
    @staticmethod
    def forward(self: Any, input_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:  # type: ignore[override]
        self.batch_size = batch_size
        return input_tensor

    @staticmethod
    def backward(self: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:  # type: ignore[override]
        grad_input = grad_output.clone()
        return grad_input * self.batch_size, None


# Scale gradients in the backward pass
# https://github.com/jcjohnson/neural-style/issues/450
# https://github.com/ProGamerGov/neural-style-pt/blob/cbcd023326a3487a2d75270ed1f3b3ddb4b72407/neural_style.py#L404
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(self: Any, input_tensor: torch.Tensor, weight: float) -> torch.Tensor:  # type: ignore[override]
        self.weight = weight
        return input_tensor

    @staticmethod
    def backward(self: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:  # type: ignore[override]
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.weight, None


class FeatureReconstructionOperator(ops.FeatureReconstructionOperator):
    r"""Feature reconstruction operator from :cite:`ULVL2016,UVL2017`.

    Args:
        encoder: Encoder used to encode the input.
        impl_params: If ``True``, normalize the score twice by the batch size.
        **feature_reconstruction_op_kwargs: Additional parameters of a
            :class:`pystiche.ops.FeatureReconstructionOperator`.

    .. seealso::

        - :class:`pystiche.ops.FeatureReconstructionOperator`
    """

    def __init__(
        self,
        encoder: enc.Encoder,
        impl_params: bool = True,
        **feature_reconstruction_op_kwargs: Any,
    ) -> None:
        super().__init__(encoder, **feature_reconstruction_op_kwargs)
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

        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L217
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L162
        # nn.MSECriterion() was used to calculate the content loss, which by default
        # uses reduction="mean" which also includes the batch_size. However, the
        # score is divided once more by the batch_size in the reference implementation.
        batch_size = image.extract_batch_size(input_repr)
        score = RemoveBatchSizeDivisionGradient.apply(score, batch_size)
        return cast(torch.Tensor, score / batch_size)

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        input_image = AddScoreWeightGradient.apply(input_image, self.score_weight)
        score = self.process_input_image(input_image)
        score = RemoveScoreWeightGradient.apply(score, self.score_weight)
        return cast(Union[torch.Tensor, pystiche.LossDict], score * self.score_weight)


def content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> FeatureReconstructionOperator:
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

    return FeatureReconstructionOperator(
        multi_layer_encoder.extract_encoder(hyper_parameters.content_loss.layer),
        impl_params=impl_params,
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
        self, encoder: enc.Encoder, impl_params: bool = True, **gram_op_kwargs: Any,
    ):
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L38
        # In the reference implementation the gram_matrix is only divided by the
        # batch_size.
        self.normalize_by_num_channels = impl_params
        super().__init__(encoder, normalize=False, **gram_op_kwargs)

        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L56-L57
        # In the reference implementation the gradients of the style_loss are
        # normalized. For the transfer from luatorch to pytorch see this discussion:
        # https://github.com/jcjohnson/neural-style/issues/450
        self.normalize_gradients = impl_params

        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L217
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L162
        # nn.MSECriterion() was used to calculate the style loss, which by default uses
        # uses reduction="mean" which also includes the batch_size. However, the
        # score is divided once more by the batch_size in the reference implementation.
        self.double_batch_size_mean = impl_params

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        if self.normalize_gradients:
            enc = ScaleGradients.apply(enc, self.score_weight)

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

    def forward(
        self, input_image: torch.Tensor
    ) -> Union[torch.Tensor, pystiche.LossDict]:
        input_image = AddScoreWeightGradient.apply(input_image, self.score_weight)
        score = self.process_input_image(input_image)
        score = RemoveScoreWeightGradient.apply(score, self.score_weight)
        return cast(Union[torch.Tensor, pystiche.LossDict], score * self.score_weight)


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
