import pytest

from pystiche_papers import gatys_ecker_bethge_2015 as paper

from .asserts import assert_image_downloads_correctly


@pytest.mark.slow
def test_gatys_ecker_bethge_2015_images(subtests):
    for name, image in paper.gatys_ecker_bethge_2015_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
