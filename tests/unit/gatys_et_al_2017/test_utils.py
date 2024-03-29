import pytest

from tests import utils

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.gatys_ecker_bethge_2016
import pystiche_papers.gatys_et_al_2017 as paper
from pystiche import enc
from pystiche_papers.utils import HyperParameters


def test_preprocessor():
    assert isinstance(paper.preprocessor(), enc.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), enc.CaffePostprocessing)


@pytest.mark.slow
def test_multi_layer_encoder(subtests):
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


def test_compute_layer_weights():
    multi_layer_encoder = paper.multi_layer_encoder()
    layers = tuple(dict(multi_layer_encoder.named_children()).keys())

    actual = paper.compute_layer_weights(layers)
    expected = pystiche_papers.gatys_ecker_bethge_2016.compute_layer_weights(
        layers, multi_layer_encoder=multi_layer_encoder
    )

    assert actual == pytest.approx(expected)


@utils.impl_params
def test_hyper_parameters_smoke(impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)
    assert isinstance(hyper_parameters, HyperParameters)


@utils.impl_params
def test_hyper_parameters_content_loss(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "content_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layer == "relu4_2" if impl_params else "conv4_2"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e0)


@utils.impl_params
def test_hyper_parameters_style_loss(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "style_loss"
    assert sub_params in hyper_parameters
    parameters = getattr(hyper_parameters, sub_params)

    layers, num_channels = zip(
        ("relu1_1" if impl_params else "conv1_1", 64),
        ("relu2_1" if impl_params else "conv2_1", 128),
        ("relu3_1" if impl_params else "conv3_1", 256),
        ("relu4_1" if impl_params else "conv4_1", 512),
        ("relu5_1" if impl_params else "conv5_1", 512),
    )
    layer_weights = [1 / n**2 for n in num_channels]

    with subtests.test("layers"):
        assert parameters.layers == layers

    with subtests.test("layer_weights"):
        assert parameters.layer_weights == pytest.approx(layer_weights)

    with subtests.test("score_weight"):
        assert parameters.score_weight == pytest.approx(1e3)


@utils.impl_params
def test_hyper_parameters_guided_style_loss(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "guided_style_loss"
    assert sub_params in hyper_parameters
    parameters = getattr(hyper_parameters, sub_params)

    layers, num_channels = zip(
        ("relu1_1" if impl_params else "conv1_1", 64),
        ("relu2_1" if impl_params else "conv2_1", 128),
        ("relu3_1" if impl_params else "conv3_1", 256),
        ("relu4_1" if impl_params else "conv4_1", 512),
        ("relu5_1" if impl_params else "conv5_1", 512),
    )
    layer_weights = [1 / n**2 for n in num_channels]

    with subtests.test("layers"):
        assert parameters.layers == layers

    with subtests.test("layer_weights"):
        assert parameters.layer_weights == pytest.approx(layer_weights)

    with subtests.test("region_weights"):
        assert parameters.region_weights == "sum"

    with subtests.test("score_weight"):
        assert parameters.score_weight == pytest.approx(1e3)


@utils.impl_params
def test_hyper_parameters_image_pyramid(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "image_pyramid"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_sizes"):
        assert hyper_parameters.edge_sizes == (512 if impl_params else 500, 1024)

    with subtests.test("num_steps"):
        if impl_params:
            assert hyper_parameters.num_steps == (500, 200)
        else:
            assert len(hyper_parameters.num_steps) == 2
            ratio = hyper_parameters.num_steps[0] / hyper_parameters.num_steps[1]
            assert ratio == pytest.approx(2.5)
