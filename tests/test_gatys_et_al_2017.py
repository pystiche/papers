import pytest
import torch

from pystiche_papers import gatys_et_al_2017 as paper

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable
from pystiche_papers.gatys_et_al_2017 import utils

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
    preprocessor = utils.gatys_et_al_2017_preprocessor()
    input = torch.tensor([[[1.485]], [[2.458]], [[3.408]]])
    actual = preprocessor(input)
    desired = [3 * 255.0, 2 * 255.0, 1 * 255.0]

    assert torch.flatten(actual).tolist() == desired