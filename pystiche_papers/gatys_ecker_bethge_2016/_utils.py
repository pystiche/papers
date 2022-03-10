import contextlib
from typing import Optional, Sequence, Tuple, cast

import torch
from torch import nn, optim

from pystiche import enc, meta
from pystiche_papers.utils import HyperParameters

__all__ = [
    "preprocessor",
    "postprocessor",
    "optimizer",
    "multi_layer_encoder",
    "compute_layer_weights",
    "hyper_parameters",
]


def preprocessor() -> nn.Module:
    return enc.CaffePreprocessing()


def postprocessor() -> nn.Module:
    return enc.CaffePostprocessing()


def multi_layer_encoder(impl_params: bool = True,) -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`GEB2016`.

    Args:
        impl_params: If ``True``, the :class:`~torch.nn.MaxPool2d` in
            the ``multi_layer_encoder`` are exchanged for :class:`~torch.nn.AvgPool2d`.

    """
    # TODO: check if the encoder used inplace ops
    multi_layer_encoder_ = enc.vgg19_multi_layer_encoder(
        framework="caffe", internal_preprocessing=False, allow_inplace=True
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


multi_layer_encoder_ = multi_layer_encoder


def compute_layer_weights(
    layers: Sequence[str], multi_layer_encoder: Optional[enc.MultiLayerEncoder] = None,
) -> Tuple[float, ...]:
    def find_out_channels(multi_layer_encoder: nn.Module, layer: str) -> int:
        modules = multi_layer_encoder._modules
        layers = list(modules.keys())
        layers = reversed(layers[: layers.index(layer) + 1])
        for layer_ in layers:
            with contextlib.suppress(AttributeError):
                return cast(
                    int, modules[layer_].out_channels  # type: ignore[union-attr]
                )

        raise RuntimeError(
            f"Neither '{layer}' nor any previous layer has an 'out_channels' "
            f"attribute."
        )

    if multi_layer_encoder is None:
        multi_layer_encoder = multi_layer_encoder_()

    num_channels = []
    for layer in layers:
        if layer not in multi_layer_encoder:
            raise ValueError(f"Layer {layer} is not part of the multi_layer_encoder.")

        num_channels.append(find_out_channels(multi_layer_encoder, layer))

    return tuple(1.0 / n ** 2.0 for n in num_channels)


def hyper_parameters(impl_params: bool = True) -> HyperParameters:
    r"""Hyper parameters from :cite:`GEB2016`."""
    # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
    # Cell [8]
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
    # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
    # Cell [8]
    style_loss_layer_weights = (
        compute_layer_weights(style_loss_layers) if impl_params else "mean"
    )

    return HyperParameters(
        content_loss=HyperParameters(
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [8]
            layer=f"{'relu' if impl_params else 'conv'}4_2",
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [8]
            score_weight=1e0,
        ),
        style_loss=HyperParameters(
            layers=style_loss_layers,
            layer_weights=style_loss_layer_weights,
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [8]
            score_weight=1e3,
        ),
        nst=HyperParameters(
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [9]
            num_steps=500,
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [6]
            starting_point="content" if impl_params else "random",
            # https://github.com/pmeier/PytorchNeuralStyleTransfer/blob/master/NeuralStyleTransfer.ipynb
            # Cell [4]
            image_size=512,
        ),
    )
