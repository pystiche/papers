import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche import enc, meta
from pystiche_papers.utils import HyperParameters

from tests import mocks, utils


def test_preprocessor():
    assert isinstance(paper.preprocessor(), enc.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), enc.CaffePostprocessing)


@pytest.mark.slow
def test_multi_layer_encoder(subtests, mocker):
    mocks.patch_models_load_state_dict_from_url(mocker=mocker)

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
    mocks.patch_models_load_state_dict_from_url(mocker=mocker)

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


def test_compute_layer_weights():
    out_channels = (3, 6, 6)

    modules = (
        ("conv1", nn.Conv2d(1, out_channels[0], 3)),
        ("conv2", nn.Conv2d(out_channels[0], out_channels[1], 3)),
        ("pool", nn.MaxPool2d(2)),
    )
    multi_layer_encoder = enc.MultiLayerEncoder(modules)
    layers, _ = zip(*modules)

    layer_weights = paper.compute_layer_weights(
        layers, multi_layer_encoder=multi_layer_encoder
    )
    assert layer_weights == pytest.approx([1 / n ** 2 for n in out_channels])


def test_get_layer_weights_wrong_layers(subtests):
    multi_layer_encoder = enc.MultiLayerEncoder((("relu", nn.ReLU()),))

    with subtests.test("layer not in multi_layer_encoder"):
        with pytest.raises(ValueError):
            paper.compute_layer_weights(
                ("not_included",), multi_layer_encoder=multi_layer_encoder
            )

    with subtests.test("no out_channels"):
        with pytest.raises(RuntimeError):
            paper.compute_layer_weights(
                ("relu",), multi_layer_encoder=multi_layer_encoder
            )


@utils.impl_params
def test_hyper_parameters(impl_params):
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
    hyper_parameters = getattr(hyper_parameters, sub_params)

    if impl_params:
        layers, num_channels = zip(
            ("relu1_1", 64),
            ("relu2_1", 128),
            ("relu3_1", 256),
            ("relu4_1", 512),
            ("relu5_1", 512),
        )
        layer_weights = pytest.approx([1 / n ** 2 for n in num_channels])
    else:
        layers = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
        layer_weights = "mean"

    with subtests.test("layer"):
        assert hyper_parameters.layers == layers

    with subtests.test("layer_weights"):
        assert hyper_parameters.layer_weights == layer_weights

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e3)


@utils.impl_params
def test_hyper_parameters_nst(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "nst"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("num_steps"):
        assert hyper_parameters.num_steps == 500

    with subtests.test("starting_point"):
        assert hyper_parameters.starting_point == "content" if impl_params else "random"

    with subtests.test("image_size"):
        assert hyper_parameters.image_size == 512
