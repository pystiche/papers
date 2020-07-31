import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.li_wand_2016 as paper
from pystiche import enc
from pystiche.image import transforms


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
