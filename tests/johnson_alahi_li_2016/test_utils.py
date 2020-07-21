import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

from pystiche.enc import VGGMultiLayerEncoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche_papers.johnson_alahi_li_2016 import utils


def test_johnson_alahi_li_2016_preprocessor():
    assert isinstance(utils.johnson_alahi_li_2016_preprocessor(), CaffePreprocessing)


def test_johnson_alahi_li_2016_postprocessor():
    assert isinstance(utils.johnson_alahi_li_2016_postprocessor(), CaffePostprocessing)


@pytest.mark.slow
def test_johnson_alahi_li_2016_multi_layer_encoder(subtests):
    multi_layer_encoder = utils.johnson_alahi_li_2016_multi_layer_encoder()
    assert isinstance(multi_layer_encoder, VGGMultiLayerEncoder)

    with subtests.test("internal preprocessing"):
        assert "preprocessing" not in multi_layer_encoder

    with subtests.test("inplace"):
        relu_modules = [
            module
            for module in multi_layer_encoder.modules()
            if isinstance(module, nn.ReLU)
        ]
        assert all(module.inplace for module in relu_modules)


def test_johnson_alahi_li_2016_optimizer(subtests, input_image):
    transformer = nn.Conv2d(3, 3, 1)
    params = tuple(transformer.parameters())
    optimizer = utils.johnson_alahi_li_2016_optimizer(transformer)

    assert isinstance(optimizer, optim.Adam)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == len(params)
        for actual, desired in zip(param_group["params"], params):
            assert actual is desired

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1e-3)
