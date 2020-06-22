import pytest

from pystiche_papers.data import NPRgeneral

from ..asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_ulyanov_et_al_2016_images_smoke(subtests):
    for name, image in NPRgeneral():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_gatys_et_al_2017_images(subtests):
    for name, image in NPRgeneral():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
