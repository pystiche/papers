import torch
from torch import optim

from pystiche.enc import MultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing

__all__ = ["preprocessor", "postprocessor", "multi_layer_encoder", "optimizer"]


def preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
