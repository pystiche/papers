import torch
from torch import nn, optim

from pystiche import enc, meta
from pystiche.image import transforms

__all__ = [
    "preprocessor",
    "postprocessor",
    "optimizer",
    "multi_layer_encoder",
]


def preprocessor() -> nn.Module:
    return transforms.CaffePreprocessing()


def postprocessor() -> nn.Module:
    return transforms.CaffePostprocessing()


def multi_layer_encoder(impl_params: bool = True,) -> enc.MultiLayerEncoder:
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )
    if impl_params:
        return multi_layer_encoder

    for name, module in multi_layer_encoder.named_children():
        if isinstance(module, nn.MaxPool2d):
            multi_layer_encoder._modules[name] = nn.AvgPool2d(
                **meta.pool_module_meta(module)
            )
    return multi_layer_encoder


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
