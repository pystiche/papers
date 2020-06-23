import pytest

from pystiche.image.transforms import CaffePreprocessing
from pystiche_papers import gatys_et_al_2017 as paper
from pystiche_papers.gatys_et_al_2017 import utils

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable


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
