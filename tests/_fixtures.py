import pytest

import torch
from torch import nn

from pystiche.enc import MultiLayerEncoder
from pystiche.image import make_single_image

from ._assets import read_image

__all__ = [
    "image_small_0",
    "image_small_1",
    "image_small_2",
    "image_small_landscape",
    "image_small_portrait",
    "image_medium",
    "image_large",
    "image",
    "single_image",
    "batch_image",
    "target_image",
    "input_image",
    "multi_layer_encoder_with_layer",
    "vgg_load_weights_mock",
    "load_weights_mocks",
]


@pytest.fixture
def image_small_0():
    return read_image("small_0")


@pytest.fixture
def image_small_1():
    return read_image("small_1")


@pytest.fixture
def image_small_2():
    return read_image("small_2")


@pytest.fixture
def image_small_landscape():
    return read_image("small_landscape")


@pytest.fixture
def image_small_portrait():
    return read_image("small_portrait")


@pytest.fixture
def image_medium():
    return read_image("medium")


@pytest.fixture
def image_large():
    return read_image("large")


@pytest.fixture
def image(image_small_0):
    return image_small_0


@pytest.fixture
def single_image(image_small_0):
    return make_single_image(image_small_0)


@pytest.fixture
def batch_image(image_small_0, image_small_1, image_small_2):
    return torch.cat((image_small_0, image_small_1, image_small_2))


@pytest.fixture
def target_image(image_small_0):
    return image_small_0


@pytest.fixture
def input_image(image_small_1):
    return image_small_1


@pytest.fixture(scope="session")
def multi_layer_encoder_with_layer():
    layer = "conv"
    multi_layer_encoder = MultiLayerEncoder(((layer, nn.Conv2d(3, 3, 1)),))
    return multi_layer_encoder, layer


@pytest.fixture(scope="session")
def vgg_load_weights_mock(session_mocker):
    return session_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="session")
def load_weights_mocks(vgg_load_weights_mock):
    return (vgg_load_weights_mock,)
