import torch
from torch import optim

from pystiche import enc
from pystiche.image import transforms

__all__ = ["preprocessor", "postprocessor", "multi_layer_encoder", "optimizer"]


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
