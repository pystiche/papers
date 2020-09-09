from typing import Optional, Sequence, Tuple, cast

import torch
from torch import nn, optim

from pystiche import enc, meta
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

__all__ = [
    "preprocessor",
    "postprocessor",
    "optimizer",
    "multi_layer_encoder",
    "hyper_parameters",
]


def preprocessor() -> nn.Module:
    return transforms.CaffePreprocessing()


def postprocessor() -> nn.Module:
    return transforms.CaffePostprocessing()


def multi_layer_encoder(impl_params: bool = True,) -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, the :class:`~torch.nn.MaxPool2d` in
            the ``multi_layer_encoder`` are exchanged for :class:`~torch.nn.AvgPool2d`.

    """
    multi_layer_encoder_ = enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )
    if impl_params:
        return multi_layer_encoder_

    for name, module in multi_layer_encoder_.named_children():
        if isinstance(module, nn.MaxPool2d):
            multi_layer_encoder_._modules[name] = nn.AvgPool2d(
                **meta.pool_module_meta(module)
            )
    return multi_layer_encoder_


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""Optimizer from :cite:`GEB2016`.

    Args:
        input_image: Image to be optimized.

    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


def compute_layer_weights(
    multi_layer_encoder: enc.MultiLayerEncoder, layers: Sequence[str],
) -> Tuple[float, ...]:
    nums_channels = []
    for layer in layers:
        if layer not in multi_layer_encoder:
            raise RuntimeError

        layer = layer.replace("relu", "conv")
        if not layer.startswith("conv"):
            raise RuntimeError

        module = cast(nn.Conv2d, multi_layer_encoder._modules[layer])
        nums_channels.append(module.out_channels)

    return tuple(1.0 / num_channels ** 2.0 for num_channels in nums_channels)


multi_layer_encoder_ = multi_layer_encoder


def hyper_parameters(
    impl_params: bool = True,
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> HyperParameters:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_(impl_params=impl_params)

    style_loss_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")

    return HyperParameters(
        content_loss=HyperParameters(layer="relu4_2", score_weight=1e0),
        style_loss=HyperParameters(
            layers=style_loss_layers,
            layer_weights=compute_layer_weights(multi_layer_encoder, style_loss_layers),
            score_weight=1e3,
        ),
        nst=HyperParameters(num_steps=500),
    )
