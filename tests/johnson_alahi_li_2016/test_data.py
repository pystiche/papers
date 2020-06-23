import pytest

from pystiche_papers import johnson_alahi_li_2016 as paper

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_johnson_alahi_li_2016_smoke(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_johnson_alahi_li_2016_images(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
