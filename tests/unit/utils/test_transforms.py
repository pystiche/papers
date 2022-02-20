import pytest

import pytorch_testing_utils as ptu

from pystiche_papers import utils


class TestOptionalGrayscaleToFakegrayscale:
    @pytest.fixture
    def transform(self):
        return utils.OptionalGrayscaleToFakegrayscale()

    def test_grayscale(self, transform, content_image):
        single_channel = content_image[:, :1]
        ptu.assert_allclose(
            transform(single_channel), single_channel.repeat(1, 3, 1, 1)
        )

    def test_no_grayscale(self, transform, content_image):
        ptu.assert_allclose(transform(content_image), content_image)
