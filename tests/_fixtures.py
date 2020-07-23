import pytest

import torch
from torch import nn

from pystiche.enc import MultiLayerEncoder
from pystiche.image import make_single_image

from ._assets import read_image
from ._utils import create_guides

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
    "target_guides",
    "input_image",
    "input_guides",
    "content_image",
    "content_guides",
    "style_image",
    "style_guides",
    "style_images_and_guides",
    "multi_layer_encoder_with_layer",
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
def target_guides(target_image):
    return create_guides(target_image)


@pytest.fixture
def input_image(image_small_1):
    return image_small_1


@pytest.fixture
def input_guides(input_image):
    return create_guides(input_image)


@pytest.fixture
def content_image(target_image):
    return target_image


@pytest.fixture
def content_guides(content_image):
    return create_guides(content_image)


@pytest.fixture
def style_image(image_small_2):
    return image_small_2


@pytest.fixture
def style_guides(style_image):
    return create_guides(style_image)


@pytest.fixture
def style_images_and_guides(style_image, style_guides):
    return {region: (style_image, guide) for region, guide in style_guides.items()}


@pytest.fixture(scope="session")
def multi_layer_encoder_with_layer():
    layer = "conv"
    multi_layer_encoder = MultiLayerEncoder(((layer, nn.Conv2d(3, 3, 1)),))
    return multi_layer_encoder, layer
