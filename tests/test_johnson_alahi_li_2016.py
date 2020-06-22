import pytest

from pystiche_papers import johnson_alahi_li_2016 as paper

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_ulyanov_et_al_2016_images_smoke(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_gatys_et_al_2017_images(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
