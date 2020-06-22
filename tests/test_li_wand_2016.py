import pytest

from pystiche_papers import li_wand_2016 as paper

from .asserts import assert_image_downloads_correctly


@pytest.mark.slow
def test_li_wand_2016_images(subtests):
    for name, image in paper.li_wand_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)
