import torch
from torch import nn, optim

from pystiche.enc import MultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche.meta import pool_module_meta

__all__ = [
    "preprocessor",
    "postprocessor",
    "optimizer",
    "multi_layer_encoder",
]


def preprocessor() -> nn.Module:
    return CaffePreprocessing()


def postprocessor() -> nn.Module:
    return CaffePostprocessing()


def multi_layer_encoder(impl_params: bool = True,) -> MultiLayerEncoder:
    multi_layer_encoder_ = vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )
    if impl_params:
        return multi_layer_encoder_

    for name, module in multi_layer_encoder_.named_children():
        if isinstance(module, nn.MaxPool2d):
            multi_layer_encoder_._modules[name] = nn.AvgPool2d(
                **pool_module_meta(module)
            )
    return multi_layer_encoder_


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
