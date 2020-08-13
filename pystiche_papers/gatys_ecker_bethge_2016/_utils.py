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
    r"""Multi-layer encoder from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    If ``impl_params is True`` the :class:`~torch.nn.MaxPool2d` in the
    ``multi_layer_encoder`` are exchanged for :class:`~torch.nn.AvgPool2d`.

    """
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
    r"""Optimizer from :cite:`GEB2016`.

        Args:
            input_image: Image to be optimized.
        """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
