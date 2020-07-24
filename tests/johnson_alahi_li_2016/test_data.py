import itertools

import pytest

import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

from pystiche.data import ImageFolderDataset
from pystiche.image import transforms
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale, resize
from pystiche_papers import johnson_alahi_li_2016 as paper
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.johnson_alahi_li_2016 import data
from pystiche_papers.utils import make_reproducible

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_johnson_alahi_li_2016_content_transform():
    make_reproducible()
    image = torch.rand(1, 1, 16, 31)

    content_transform = data.johnson_alahi_li_2016_content_transform(impl_params=False)
    actual = content_transform(image)

    desired = grayscale_to_fakegrayscale(resize(image[:, :, :16, :16], 256))

    ptu.assert_allclose(actual, desired)


def test_johnson_alahi_li_2016_content_transform_impl_params():
    content_transform = data.johnson_alahi_li_2016_content_transform()
    preprocessing = tuple(content_transform.children())[-1]
    assert isinstance(preprocessing, transforms.CaffePreprocessing)


def test_get_style_edge_size_smoke(subtests, styles):
    for impl_params, instance_norm, style in itertools.product(
        (True, False), (True, False), styles
    ):
        if not impl_params and instance_norm:
            continue

        with subtests.test(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        ):
            assert isinstance(
                data.get_style_edge_size(impl_params, instance_norm, style=style), int,
            )


def test_johnson_alahi_li_2016_style_transform(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        if not impl_params and instance_norm:
            continue

        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            style_transform = data.johnson_alahi_li_2016_style_transform(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(style_transform, transforms.Resize)

            with subtests.test("edge_size"):
                edge_size = data.get_style_edge_size(impl_params, instance_norm)
                assert style_transform.size == edge_size

            with subtests.test("edge"):
                assert style_transform.edge == "long"


@pytest.mark.slow
def test_johnson_alahi_li_2016_images_smoke(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_johnson_alahi_li_2016_images(subtests):
    for name, image in paper.johnson_alahi_li_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_johnson_alahi_li_2016_dataset(subtests, mocker):
    mocker.patch(
        "pystiche.data.datasets.ImageFolderDataset._collect_image_files",
        return_value=[],
    )
    dataset = data.johnson_alahi_li_2016_dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("transform"):
        assert isinstance(
            dataset.transform, type(data.johnson_alahi_li_2016_content_transform())
        )


def test_johnson_alahi_li_2016_batch_sampler(subtests):
    data_source = ()
    batch_sampler = data.johnson_alahi_li_2016_batch_sampler(data_source)

    assert isinstance(batch_sampler, FiniteCycleBatchSampler)

    with subtests.test("num_batches"):
        assert batch_sampler.num_batches == 40000

    with subtests.test("num_batches"):
        assert batch_sampler.batch_size == 4


def test_johnson_alahi_li_2016_image_loader(subtests):
    dataset = ()
    image_loader = data.johnson_alahi_li_2016_image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler,
            type(data.johnson_alahi_li_2016_batch_sampler(dataset)),
        )
