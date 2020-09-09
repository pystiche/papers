import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche import enc, meta
from pystiche.image import transforms

from tests import mocks


def test_preprocessor():
    assert isinstance(paper.preprocessor(), transforms.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), transforms.CaffePostprocessing)


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

    sub_params = "style_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    layers, num_channels = zip(
        ("relu1_1", 64),
        ("relu2_1", 128),
        ("relu3_1", 256),
        ("relu4_1", 512),
        ("relu5_1", 512),
    )

    with subtests.test("layer"):
        assert hyper_parameters.layers == layers

    with subtests.test("layer_weights"):
        assert hyper_parameters.layer_weights == pytest.approx(
            [1 / n ** 2 for n in num_channels]
        )

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e3)


def test_hyper_parameters_nst(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "nst"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("num_steps"):
        assert hyper_parameters.num_steps == 500


# def test_get_layer_weights():
#     layers, nums_channels = zip(
#         ("relu1_1", 64),
#         ("relu2_1", 128),
#         ("relu3_1", 256),
#         ("relu4_1", 512),
#         ("relu5_1", 512),
#     )
#
#     actual = get_layer_weights(layers)
#     desired = tuple(1.0 / num_channels ** 2.0 for num_channels in nums_channels)
#     assert actual == pytest.approx(desired)
#
#
# def test_get_layer_weights_wrong_layers(subtests):
#     with subtests.test("layer not in multi_layer_encoder"):
#         not_included_layers = ("not_included",)
#
#         with pytest.raises(RuntimeError):
#             get_layer_weights(not_included_layers)
#
#     with subtests.test("no conv or relu layer"):
#         no_conv_or_relu_layers = ("pool1",)
#
#         with pytest.raises(RuntimeError):
#             get_layer_weights(no_conv_or_relu_layers)
#
# def test_style_loss_wrong_layers(mocker):
#     mock = mocker.patch("pystiche_papers.gatys_ecker_bethge_2016._loss.StyleLoss")
#
#     layers = ("not_included", "not_conv_or_relu")
#
#     with pytest.warns(RuntimeWarning):
#         paper.style_loss(layers=layers)
#
#     assert mock.call_args[1]["layer_weights"] == "mean"
