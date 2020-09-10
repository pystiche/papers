import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.gatys_ecker_bethge_2016
import pystiche_papers.gatys_et_al_2017 as paper
from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters


def test_preprocessor():
    assert isinstance(paper.preprocessor(), transforms.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), transforms.CaffePostprocessing)


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
    layers = tuple(multi_layer_encoder.children_names())

    actual = paper.compute_layer_weights(layers)
    expected = pystiche_papers.gatys_ecker_bethge_2016.compute_layer_weights(
        layers, multi_layer_encoder=multi_layer_encoder
    )

    assert actual == pytest.approx(expected)


def test_hyper_parameters_smoke(subtests):
    hyper_parameters = paper.hyper_parameters()
    assert isinstance(hyper_parameters, HyperParameters)


def test_hyper_parameters_content_loss(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "content_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layer == "relu4_2"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e0)


def test_hyper_parameters_style_loss(subtests):
    hyper_parameters = paper.hyper_parameters()

    for sub_params in ("style_loss", "guided_style_loss"):
        with subtests.test(sub_params):

            assert sub_params in hyper_parameters
            parameters = getattr(hyper_parameters, sub_params)

            layers, num_channels = zip(
                ("relu1_1", 64),
                ("relu2_1", 128),
                ("relu3_1", 256),
                ("relu4_1", 512),
                ("relu5_1", 512),
            )

            with subtests.test("layer"):
                assert parameters.layers == layers

            with subtests.test("layer_weights"):
                assert parameters.layer_weights == pytest.approx(
                    [1 / n ** 2 for n in num_channels]
                )

            with subtests.test("score_weight"):
                assert parameters.score_weight == pytest.approx(1e3)

    with subtests.test("guided_style_loss, region_weights"):
        assert hyper_parameters.guided_style_loss.region_weights == "sum"


def test_hyper_parameters_image_pyramid(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "image_pyramid"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_sizes"):
        assert hyper_parameters.edge_sizes == (500, 800)

    with subtests.test("num_steps"):
        assert hyper_parameters.num_steps == (500, 200)
