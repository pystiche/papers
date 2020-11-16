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

        # instance_norm:
        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L217
        # not instance_norm:
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L162
        # nn.MSECriterion() was used to calculate the content loss, which by default
        # uses reduction="mean" which also includes the batch_size. However, the
        # score is divided once more by the batch_size in the reference implementation.
        batch_size = image.extract_batch_size(input_repr)
        return score / batch_size


def content_loss(
    impl_params: bool = True,
    instance_norm: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layer: str = "relu4_2",
    score_weight: Optional[float] = None,
) -> FeatureReconstructionOperator:
    r"""Content_loss from :cite:`ULVL2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>` and below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.multi_layer_encoder` is used.
        layer: Layer from which the encodings of the ``multi_layer_encoder`` should be
            taken. Defaults to ``"relu4_2"``.
        score_weight: Score weight of the operator. If omitted, the ``score_weight`` is
            determined with respect to ``impl_params`` and ``instance_norm``. For
            details see :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.

    If ``impl_params is True`` , the score is divided twice by the batch_size.

    """
    if score_weight is None:
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L22
        score_weight = 6e-1 if impl_params and not instance_norm else 1e0

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
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/src/texture_loss.lua#L38
        self.normalize_by_num_channels = impl_params
        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L217
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L162
        # nn.MSECriterion() was used to calculate the style loss, which by default uses
        # uses reduction="mean" which also includes the batch_size. However, the
        # score is divided once more by the batch_size in the reference implementation.
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
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    score_weight: Optional[float] = None,
    **gram_op_kwargs: Any,
) -> ops.MultiLayerEncodingOperator:
    r"""Style_loss from :cite:`ULVL2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>` and below.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.ulyanov_et_al_2016.multi_layer_encoder` is used.
        layers: Layers from which the encodings of the ``multi_layer_encoder`` should be
            taken. If omitted, the layers are determined with respect to ``impl_params``
            and ``instance_norm``. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        layer_weights: Layer weights of the operator. Defaults to ``"sum"``.
        score_weight: Score weight of the operator. If omitted, the ``score_weight`` is
            determined with respect to ``instance_norm`` and ``impl_params``. For
            details see :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        **gram_op_kwargs: Optional parameters for the
            :class:`~pystiche.ops.GramOperator`.

    If ``impl_params`` is ``True`` , the score is divided twice by the batch_size.

    """
    if score_weight is None:
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_train.lua#L23
        score_weight = 1e3 if impl_params and not instance_norm else 1e0

    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if layers is None:
        if impl_params and instance_norm:
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/train.lua#L44
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
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
) -> loss.PerceptualLoss:
    r"""Perceptual loss from :cite:`ULVL2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        multi_layer_encoder: Pretrained :class:`~pystiche.enc.MultiLayerEncoder`. If
            omitted, the default
            :func:`~pystiche_papers.johnson_alahi_li_2016._utils.multi_layer_encoder`
            is used.
        content_loss_kwargs: Optional parameters for the :func:`content_loss`.
        style_loss_kwargs: Optional parameters for the :func:`style_loss`.

    """
    if multi_layer_encoder is None:
        multi_layer_encoder = _multi_layer_encoder()

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss_ = style_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss_ = content_loss(
        impl_params=impl_params,
        instance_norm=instance_norm,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    return loss.PerceptualLoss(content_loss_, style_loss_)
