import itertools

import pytest

import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

import pystiche.image.transforms.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche.data import ImageFolderDataset
from pystiche.image import transforms
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.utils import make_reproducible

from tests.asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_content_transform():
    make_reproducible()
    image = torch.rand(1, 1, 16, 31)

    content_transform = paper.content_transform(impl_params=False)
    actual = content_transform(image)

    desired = F.grayscale_to_fakegrayscale(F.resize(image[:, :, :16, :16], 256))

    ptu.assert_allclose(actual, desired)


def test_content_transform_impl_params():
    content_transform = paper.content_transform()
    preprocessing = tuple(content_transform.children())[-1]
    assert isinstance(preprocessing, transforms.CaffePreprocessing)


def test_style_transform(subtests, styles):
    for impl_params, instance_norm, style in itertools.product(
        (True, False), (True, False), styles
    ):

        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            style_transform = paper.style_transform(
                impl_params=impl_params, instance_norm=instance_norm, style=style
            )

            assert isinstance(style_transform, transforms.Resize)

            with subtests.test("edge"):
                assert style_transform.edge == "long"


def test_style_transform_edge_size_default(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            style_transform = paper.style_transform(
                impl_params=impl_params, instance_norm=instance_norm
            )
            assert style_transform.size == 256


def test_style_transform_edge_size_luatorch(subtests):
    configs = (
        ("candy", True, 384),
        ("composition_vii", False, 512),
        ("feathers", True, 180),
        ("la_muse", False, 512),
        ("la_muse", True, 512),
        ("mosaic", True, 512),
        ("starry_night", False, 512),
        ("the_scream", True, 384),
        ("the_wave", False, 512),
        ("udnie", True, 256),
    )

    for style, instance_norm, edge_size in configs:
        with subtests.test(style=style, instance_norm=instance_norm):
            style_transform = paper.style_transform(
                impl_params=True, instance_norm=instance_norm, style=style
            )
            assert style_transform.size == edge_size


@pytest.mark.slow
def test_images_smoke(subtests):
    for name, image in paper.images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_images(subtests):
    for name, image in paper.images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_dataset(subtests, mocker):
    mocker.patch(
        "pystiche_papers.johnson_alahi_li_2016._data.ImageFolderDataset._collect_image_files",
        return_value=[],
    )
    dataset = paper.dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("transform"):
        assert isinstance(dataset.transform, type(paper.content_transform()))


def test_batch_sampler(subtests):
    batch_sampler = paper.batch_sampler(())

    assert isinstance(batch_sampler, FiniteCycleBatchSampler)

    with subtests.test("num_batches"):
        assert batch_sampler.batch_size == 4


def test_batch_sampler_num_batches_default(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            batch_sampler = paper.batch_sampler(
                (), impl_params=impl_params, instance_norm=instance_norm
            )
            assert batch_sampler.num_batches == 40000


def test_batch_sampler_num_batches_luatorch(subtests):
    configs = (
        ("candy", True, 40000),
        ("composition_vii", False, 60000),
        ("feathers", True, 60000),
        ("la_muse", False, 40000),
        ("la_muse", True, 40000),
        ("mosaic", True, 60000),
        ("starry_night", False, 40000),
        ("the_scream", True, 60000),
        ("the_wave", False, 40000),
        ("udnie", True, 40000),
    )

    for style, instance_norm, num_batches in configs:
        with subtests.test(style=style, instance_norm=instance_norm):
            batch_sampler = paper.batch_sampler(
                (), impl_params=True, instance_norm=instance_norm, style=style
            )
            assert batch_sampler.num_batches == num_batches


def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler, type(paper.batch_sampler(dataset)),
        )
