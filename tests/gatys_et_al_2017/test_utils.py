import pytest

from torch import optim

from pystiche.enc import VGGMultiLayerEncoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche_papers.gatys_et_al_2017 import utils


def test_gatys_et_al_2017_preprocessor():
    assert isinstance(utils.gatys_et_al_2017_preprocessor(), CaffePreprocessing)


def test_gatys_et_al_2017_postprocessor():
    assert isinstance(utils.gatys_et_al_2017_postprocessor(), CaffePostprocessing)


@pytest.mark.large_download
@pytest.mark.slow
def test_gatys_et_al_2017_multi_layer_encoder():
    assert isinstance(
        utils.gatys_et_al_2017_multi_layer_encoder(), VGGMultiLayerEncoder
    )


def test_gatys_et_al_2017_optimizer(input_image):
    optimizer = utils.gatys_et_al_2017_optimizer(input_image)
    assert isinstance(optimizer, optim.LBFGS)
    for param_group in optimizer.param_groups:
        assert param_group["lr"] == 1.0
        assert param_group["max_iter"] == 1
