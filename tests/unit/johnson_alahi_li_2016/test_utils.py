import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import enc
from pystiche.image import transforms


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
