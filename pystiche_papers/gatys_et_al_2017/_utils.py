from typing import Optional, Sequence, Tuple

import torch
from torch import optim

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.gatys_ecker_bethge_2016 import (
    compute_layer_weights as _compute_layer_weights,
)
from pystiche_papers.utils import HyperParameters

__all__ = [
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
    "compute_layer_weights",
    "hyper_parameters",
]


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`GEB+2017`."""
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""Optimizer from :cite:`GEB+2017`.

    Args:
        input_image: Image to be optimized.

    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
    layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    return _compute_layer_weights(layers, multi_layer_encoder=multi_layer_encoder)


def hyper_parameters() -> HyperParameters:
    r"""Hyper parameters from :cite:`GEB+2017`."""
    style_loss_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    style_loss = HyperParameters(
        layers=style_loss_layers,
        layer_weights=compute_layer_weights(style_loss_layers),
        score_weight=1e3,
    )

    return HyperParameters(
        content_loss=HyperParameters(layer="relu4_2", score_weight=1e0),
        style_loss=style_loss,
        guided_style_loss=style_loss.new_similar(region_weights="sum"),
        image_pyramid=HyperParameters(edge_sizes=(500, 800), num_steps=(500, 200)),
    )
