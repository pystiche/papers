import pytest

from pystiche_papers import gatys_et_al_2017 as paper

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
