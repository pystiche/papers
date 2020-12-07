import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import enc
from pystiche.image import transforms


def test_hyper_parameters_content_loss(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "content_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layer == "relu2_2"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e0)


def test_hyper_parameters_style_loss(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "style_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layers == ("relu1_2", "relu2_2", "relu3_3", "relu4_3")

    with subtests.test("layer_weights"):
        assert hyper_parameters.layer_weights == "sum"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(5e0)


def test_hyper_parameters_regularization(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "regularization"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e-6)


def test_hyper_parameters_content_transform(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "content_transform"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_size"):
        assert hyper_parameters.edge_size == 256


def test_hyper_parameters_style_transform(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "style_transform"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_size"):
        assert hyper_parameters.edge_size == 256

    with subtests.test("edge"):
        assert hyper_parameters.edge == "long"


def test_hyper_parameters_batch_sampler(subtests):
    hyper_parameters = paper.hyper_parameters()

    sub_params = "batch_sampler"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("num_batches"):
        assert hyper_parameters.num_batches == 40000

    with subtests.test("batch_size"):
        assert hyper_parameters.batch_size == 4


def test_preprocessor():
    assert isinstance(
        paper.preprocessor(impl_params=True), transforms.CaffePreprocessing
    )


def test_preprocessor_noop(input_image):
    preprocessor = paper.preprocessor(impl_params=False)
    assert isinstance(preprocessor, nn.Module)
    ptu.assert_allclose(preprocessor(input_image), input_image)


def test_postprocessor():
    assert isinstance(
        paper.postprocessor(impl_params=True), transforms.CaffePostprocessing
    )


def test_postprocessor_noop(input_image):
    postprocessor = paper.postprocessor(impl_params=False)
    assert isinstance(postprocessor, nn.Module)
    ptu.assert_allclose(postprocessor(input_image), input_image)


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
    transformer = nn.Conv2d(3, 3, 1)
    params = tuple(transformer.parameters())
    optimizer = paper.optimizer(transformer)

    assert isinstance(optimizer, optim.Adam)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == len(params)
        for actual, desired in zip(param_group["params"], params):
            assert actual is desired

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1e-3)
