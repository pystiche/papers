import pytest

import pystiche_papers.gatys_et_al_2017 as paper

from .asserts import assert_image_downloads_correctly


@pytest.mark.slow
def test_images(subtests):
    for name, image in paper.images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
