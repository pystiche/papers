import pytorch_testing_utils as ptu
import torch

import pystiche.image.transforms.functional as F
from pystiche import image
from pystiche_papers import utils


def test_TopLeftCropToMultiple(content_image):
    multiple = 16
    transform = utils.TopLeftCropToMultiple(multiple)
    actual = transform(content_image)

    old_height, old_width = image.extract_image_size(content_image)
    new_height = old_height - old_height % multiple
    new_width = old_width - old_width % multiple

    desired = F.top_left_crop(content_image, (new_height, new_width))
    ptu.assert_allclose(actual, desired)


def test_OptionalGrayscaleToFakegrayscale_no_grayscale(content_image):
    transform = utils.OptionalGrayscaleToFakegrayscale()
    actual = transform(content_image)
    ptu.assert_allclose(actual, content_image)


def test_OptionalGrayscaleToFakegrayscale():
    image = torch.rand((1, 1, 32, 32))
    transform = utils.OptionalGrayscaleToFakegrayscale()
    actual = transform(image)
    desired = F.grayscale_to_fakegrayscale(image)
    ptu.assert_allclose(actual, desired)


def test_MirrorHorizontally(content_image):
    transform = utils.MirrorHorizontally()
    actual = transform(content_image)
    desired = content_image.flip(2)
    ptu.assert_allclose(actual, desired)
