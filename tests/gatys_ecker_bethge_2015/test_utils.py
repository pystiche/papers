import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.gatys_ecker_bethge_2015 as paper
from pystiche import enc, meta
from pystiche.image import transforms


def test_preprocessor():
    assert isinstance(paper.preprocessor(), transforms.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), transforms.CaffePostprocessing)


@pytest.mark.slow
def test_multi_layer_encoder(subtests, mocker):
    mocker.patch("pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights")

    multi_layer_encoder = paper.multi_layer_encoder()
    assert isinstance(multi_layer_encoder, enc.VGGMultiLayerEncoder)

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
def test_multi_layer_encoder_avg_pool(mocker):
    mocker.patch("pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights")

    multi_layer_encoder = paper.multi_layer_encoder(impl_params=False)
    pool_modules = [
        module
        for module in multi_layer_encoder.modules()
        if meta.is_pool_module(module)
    ]
    assert all(isinstance(module, nn.AvgPool2d) for module in pool_modules)


def test_optimizer(subtests, input_image):
    params = input_image
    optimizer = paper.optimizer(params)

    assert isinstance(optimizer, optim.LBFGS)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == 1
        assert param_group["params"][0] is params

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1.0)
        assert param_group["max_iter"] == 1
