import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.image import (
    extract_batch_size,
    extract_edge_size,
    extract_image_size,
    extract_num_channels,
)
from pystiche.image.transforms.functional import rescale
from pystiche_papers import utils
from pystiche_papers.sanakoyeu_et_al_2018._augmentation import (
    DynamicSizeReflectionPad2d,
    RandomAffine,
    RandomCrop,
    RandomHSVJitter,
    RandomRescale,
    RandomRotation,
    RemoveDynamicSizePadding,
)


def assert_is_size_and_value_range_preserving(input_image, transform):
    output_image = transform(input_image)
    size_fmtstr = "The {} of the output and input image mismatch: {} != {}"

    actual = extract_batch_size(output_image)
    expected = extract_batch_size(input_image)
    assert actual == expected, size_fmtstr.format("batch sizes", actual, expected)

    actual = extract_num_channels(output_image)
    expected = extract_num_channels(input_image)
    assert actual == expected, size_fmtstr.format(
        "number of channels", actual, expected
    )

    actual = extract_image_size(output_image)
    expected = extract_image_size(input_image)
    assert actual == expected, size_fmtstr.format("image sizes", actual, expected)

    assert torch.all(
        (output_image >= 0.0) & (output_image <= 1.0)
    ), "The output image contains values outside of the interval [0.0, 1.0]"


def assert_is_noop(input_image, transform, **kwargs):
    output_image = transform(input_image)
    ptu.assert_allclose(output_image, input_image, **kwargs)


@pytest.fixture(autouse=True)
def make_reproducible():
    utils.make_reproducible(0)


def test_RandomRescale(input_image):
    factor = 0.5
    transform = RandomRescale(factor, probability=100e-2)

    actual = transform(input_image)
    expected = rescale(input_image, factor)
    ptu.assert_allclose(actual, expected)


def test_RandomRescale_noop(subtests, input_image):
    for factor, probability in ((1.0, 100e-2), (0.5, 0e-2)):
        with subtests.test(factor=factor, probability=probability):
            transform = RandomRescale(factor, probability=probability)
            assert_is_noop(input_image, transform)


def test_RandomRotation_smoke(input_image):
    degrees = 30.0
    transform = RandomRotation(degrees, probability=100e-2)
    assert_is_size_and_value_range_preserving(input_image, transform)


def test_RandomRotation_noop(subtests, input_image):
    for degrees, probability in ((0.0, 100e-2), (30.0, 0e-2)):
        with subtests.test(degrees=degrees, probability=probability):
            transform = RandomRotation(degrees, probability=probability)
            assert_is_noop(input_image, transform)


def test_RandomAffine_smoke(input_image):
    shift = 0.2
    transform = RandomAffine(shift, probability=100e-2)
    assert_is_size_and_value_range_preserving(input_image, transform)


def test_RandomAffine_noop(subtests, input_image):
    for shift, probability in ((0.0, 100e-2), (0.2, 0e-2)):
        with subtests.test(degrees=shift, probability=probability):
            transform = RandomAffine(shift, probability=probability)
            assert_is_noop(input_image, transform)


def test_RandomCrop_smoke(input_image):
    size = extract_image_size(input_image)
    transform = RandomCrop(size)
    assert_is_size_and_value_range_preserving(input_image, transform)


def test_RandomCrop_size(input_image):
    size = extract_edge_size(input_image) // 2
    transform = RandomCrop(size)

    output_image = transform(input_image)
    assert extract_edge_size(output_image) == size


def test_RandomHSVJitter_smoke(input_image):
    scale = shift = 0.2
    transform = RandomHSVJitter(
        hue_scale=scale,
        hue_shift=shift,
        saturation_scale=scale,
        saturation_shift=shift,
        value_scale=scale,
        value_shift=shift,
    )
    assert_is_size_and_value_range_preserving(input_image, transform)


def test_DynamicSizePadding_noop(input_image):
    factor = 0.5
    transform = DynamicSizeReflectionPad2d(factor)
    inverse_transform = RemoveDynamicSizePadding(factor)
    assert_is_noop(
        input_image, lambda image: inverse_transform(transform(image)), atol=1e-3
    )


def test_augmentation_smoke(subtests, input_image):
    size = extract_image_size(input_image)
    probability = 100e-2
    same_on_batch = True

    input_image = utils.batch_up_image(input_image, 2)

    utils.make_reproducible()
    transform = paper.augmentation(
        size=size, probability=probability, same_on_batch=same_on_batch
    )

    output_image = transform(input_image)

    with subtests.test("size"):
        assert output_image.size() == input_image.size()

    with subtests.test("same_on_batch"):
        samples = torch.split(output_image, 1)
        for sample in samples[1:]:
            ptu.assert_allclose(sample, samples[0])

    with subtests.test("value_range"):
        assert torch.all((output_image >= 0.0) & (output_image <= 1.0))


def test_augmentation_repr_smoke():
    assert isinstance(repr(paper.augmentation()), str)
