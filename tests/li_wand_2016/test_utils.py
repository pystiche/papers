import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

from pystiche.enc import VGGMultiLayerEncoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche_papers.li_wand_2016 import utils


def test_li_wand_2016_preprocessor():
    assert isinstance(utils.li_wand_2016_preprocessor(), CaffePreprocessing)


def test_li_wand_2016_postprocessor():
    assert isinstance(utils.li_wand_2016_postprocessor(), CaffePostprocessing)


@pytest.mark.slow
def test_li_wand_2016_multi_layer_encoder(subtests, mocker):
    mocker.patch("pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights")

    multi_layer_encoder = utils.li_wand_2016_multi_layer_encoder()
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


def test_li_wand_2016_optimizer(subtests, input_image):
    params = input_image
    optimizer = utils.li_wand_2016_optimizer(params)

    assert isinstance(optimizer, optim.LBFGS)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == 1
        assert param_group["params"][0] is params

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1.0)
        assert param_group["max_iter"] == 1
