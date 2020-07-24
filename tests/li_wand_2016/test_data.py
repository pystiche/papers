import pytest

from pystiche_papers import li_wand_2016 as paper

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


@pytest.mark.slow
def test_li_wand_2016_images_smoke(subtests):
    for name, image in paper.li_wand_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_li_wand_2016_images(subtests):
    for name, image in paper.li_wand_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
