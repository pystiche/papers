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
    r"""Multi-layer encoder based on the VGG19 architecture with the weights of caffe,
    no internal preprocessing and allowed inplace.

    """
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""
        Args:
            input_image: Image to be optimized.
        Returns:
            :class:`torch.optim.LBFGS` optimizer with a learning rate of ``1.0``. The
            pixels of ``input_image`` are set as optimization parameters.
        """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
