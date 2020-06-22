import pytest

from pystiche_papers import gatys_et_al_2017 as paper

from .asserts import assert_image_downloads_correctly


@pytest.mark.slow
def test_gatys_et_al_2017_images(subtests):
    for name, image in paper.gatys_et_al_2017_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
