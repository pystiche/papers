import pytest

from pystiche_papers import data

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_NPRgeneralLicense_smoke():
    _, image = next(iter(data.NPRgeneral()))
    assert isinstance(repr(image.license), str)


def test_NPRgeneral_smoke(subtests):
    for name, image in data.NPRgeneral():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_NPRgeneral_images(subtests):
    for name, image in data.NPRgeneral():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
