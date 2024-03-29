from typing import Optional, Sequence, Tuple

import torch
from torch import optim

from pystiche import enc
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


def preprocessor() -> enc.CaffePreprocessing:
    return enc.CaffePreprocessing()


def postprocessor() -> enc.CaffePostprocessing:
    return enc.CaffePostprocessing()


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`GEB+2017`."""
    return enc.vgg19_multi_layer_encoder(
        framework="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""Optimizer from :cite:`GEB+2017`.

    Args:
        input_image: Image to be optimized.

    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
    layers: Sequence[str],
    multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()
    return _compute_layer_weights(layers, multi_layer_encoder=multi_layer_encoder)


def hyper_parameters(impl_params: bool = True) -> HyperParameters:
    r"""Hyper parameters from :cite:`GEB+2017`."""
    # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
    # Cell [3] / layers['style']
    style_loss_layers: Tuple[str, ...] = (
        "conv1_1",
        "conv2_1",
        "conv3_1",
        "conv4_1",
        "conv5_1",
    )
    if impl_params:
        style_loss_layers = tuple(
            layer.replace("conv", "relu") for layer in style_loss_layers
        )
    style_loss = HyperParameters(
        layers=style_loss_layers,
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
        # Cell [3] / weights['style']
        layer_weights=compute_layer_weights(style_loss_layers),
        # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
        # Cell [3] / sw
        score_weight=1e3,
    )

    return HyperParameters(
        content_loss=HyperParameters(
            # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
            # Cell [3] / layers['content']
            layer="relu4_2" if impl_params else "conv4_2",
            # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
            # Cell [3] / cw
            score_weight=1e0,
        ),
        style_loss=style_loss,
        guided_style_loss=style_loss.new_similar(
            # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/SpatialControl.ipynb
            # TODO: find the cell where this is performed
            region_weights="sum"
        ),
        image_pyramid=HyperParameters(
            # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
            # Cell [3] / img_size, hr_img_size
            edge_sizes=(512 if impl_params else 500, 1024),
            # https://github.com/pmeier/NeuralImageSynthesis/blob/cced0b978fe603569033b2c7f04460839e4d82c4/ExampleNotebooks/BasicStyleTransfer.ipynb
            # Cell [3] / max_iter, hr_max_iter
            num_steps=(500, 200),
        ),
    )
