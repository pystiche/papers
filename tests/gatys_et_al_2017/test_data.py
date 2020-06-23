import pytest

from torch import optim

from pystiche.enc.models.vgg import VGGMultiLayerEncoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche_papers import gatys_et_al_2017 as paper
from pystiche_papers.gatys_et_al_2017 import utils

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_gatys_et_al_2017_images_smoke(subtests):
    for name, image in paper.gatys_et_al_2017_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_gatys_et_al_2017_images(subtests):
    for name, image in paper.gatys_et_al_2017_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_gatys_et_al_2017_preprocessor():
    assert isinstance(utils.gatys_et_al_2017_preprocessor(), CaffePreprocessing)


def test_gatys_et_al_2017_postprocessor():
    assert isinstance(utils.gatys_et_al_2017_postprocessor(), CaffePostprocessing)


def test_gatys_et_al_2017_multi_layer_encoder():
    assert isinstance(
        utils.gatys_et_al_2017_multi_layer_encoder(), VGGMultiLayerEncoder
    )


def test_gatys_et_al_2017_optimizer(input_image):
    optimizer = utils.gatys_et_al_2017_optimizer(input_image)
    assert isinstance(optimizer, optim.LBFGS)
    assert optimizer.lr == 1.0
    assert optimizer.max_iter == 1
