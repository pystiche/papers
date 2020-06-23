import pytest

import torch
from torch import nn

from pystiche.enc import MultiLayerEncoder

__all__ = [
    "multi_layer_encoder_with_layer",
    "input_image",
    "target_image",
    "multi_target_image",
    "multi_input_image",
]


@pytest.fixture(scope="session")
def multi_layer_encoder_with_layer():
    layer = "conv"
    multi_layer_encoder = MultiLayerEncoder(((layer, nn.Conv2d(3, 3, 1)),))
    return multi_layer_encoder, layer


@pytest.fixture(scope="session")
def input_image():
    torch.manual_seed(hash("input_image"))
    return torch.rand((1, 3, 32, 32))


@pytest.fixture(scope="session")
def target_image():
    torch.manual_seed(hash("target_image"))
    return torch.rand((1, 3, 32, 32))


@pytest.fixture(scope="session")
def multi_target_image():
    torch.manual_seed(hash("multi_target_image"))
    return torch.rand((2, 3, 32, 32))


@pytest.fixture(scope="session")
def multi_input_image():
    torch.manual_seed(hash("multi_input_image"))
    return torch.rand((2, 3, 32, 32))
