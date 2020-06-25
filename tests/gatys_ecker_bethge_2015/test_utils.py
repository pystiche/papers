import pytest

from torch import nn, optim

import pytorch_testing_utils as ptu
from pystiche import meta
from pystiche.enc import VGGMultiLayerEncoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche_papers.gatys_ecker_bethge_2015 import utils


def test_gatys_ecker_bethge_2015_preprocessor():
    assert isinstance(utils.gatys_ecker_bethge_2015_preprocessor(), CaffePreprocessing)


def test_gatys_ecker_bethge_2015_postprocessor():
    assert isinstance(
        utils.gatys_ecker_bethge_2015_postprocessor(), CaffePostprocessing
    )


@pytest.mark.slow
def test_gatys_ecker_bethge_2015_multi_layer_encoder(subtests, mocker):
    mocker.patch("pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights")

    multi_layer_encoder = utils.gatys_ecker_bethge_2015_multi_layer_encoder()
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


@pytest.mark.slow
def test_gatys_ecker_bethge_2015_multi_layer_encoder_avg_ppol(mocker):
    mocker.patch("pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights")

    multi_layer_encoder = utils.gatys_ecker_bethge_2015_multi_layer_encoder(
        impl_params=False
    )
    pool_modules = [
        module
        for module in multi_layer_encoder.modules()
        if meta.is_pool_module(module)
    ]
    assert all(isinstance(module, nn.AvgPool2d) for module in pool_modules)


def test_gatys_ecker_bethge_2015_optimizer(subtests, input_image):
    params = input_image
    optimizer = utils.gatys_ecker_bethge_2015_optimizer(params)

    assert isinstance(optimizer, optim.LBFGS)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == 1
        assert param_group["params"][0] is params

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1.0)
        assert param_group["max_iter"] == 1
