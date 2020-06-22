import pytest

from pystiche_papers import ulyanov_et_al_2016 as paper

from .asserts import assert_image_downloads_correctly


@pytest.mark.slow
def test_ulyanov_et_al_2016_images(subtests):
    for name, image in paper.ulyanov_et_al_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
