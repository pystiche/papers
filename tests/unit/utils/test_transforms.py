import pytorch_testing_utils as ptu

import pystiche.image.transforms.functional as F
from pystiche_papers import utils


def test_OptionalGrayscaleToFakegrayscale_no_grayscale(content_image):
    transform = utils.OptionalGrayscaleToFakegrayscale()
    actual = transform(content_image)
    ptu.assert_allclose(actual, content_image)


def test_OptionalGrayscaleToFakegrayscale(content_image):
    image = F.rgb_to_grayscale(content_image)
    transform = utils.OptionalGrayscaleToFakegrayscale()
    actual = transform(image)
    desired = F.grayscale_to_fakegrayscale(image)
    ptu.assert_allclose(actual, desired)
