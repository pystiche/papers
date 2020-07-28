import itertools

import pytest

import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

from pystiche.data import ImageFolderDataset
from pystiche.image import transforms
from pystiche.image.transforms import CenterCrop, ValidRandomCrop
from pystiche.image.transforms.functional import grayscale_to_fakegrayscale, resize
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.ulyanov_et_al_2016 import data
from pystiche_papers.utils import make_reproducible

from .._asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_ulyanov_et_al_2016_content_transform(subtests):
    make_reproducible()
    image = torch.rand(1, 1, 32, 32)
    edge_size = 16

    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            make_reproducible()
            content_transform = data.ulyanov_et_al_2016_content_transform(
                edge_size=edge_size,
                impl_params=impl_params,
                instance_norm=instance_norm,
            )
            actual = content_transform(image)

            if impl_params:
                if instance_norm:
                    make_reproducible()
                    transform = ValidRandomCrop(edge_size)
                    transform_image = transform(image)
                else:
                    transform_image = resize(image, edge_size)
            else:
                transform = CenterCrop(edge_size)
                transform_image = transform(image)

            desired = grayscale_to_fakegrayscale(transform_image)

            ptu.assert_allclose(actual, desired)


def test_ulyanov_et_al_2016_style_transform(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            style_transform = data.ulyanov_et_al_2016_style_transform(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(style_transform, transforms.ComposedTransform)

            with subtests.test("edge_size"):
                assert tuple(style_transform.children())[0].size == 256

            with subtests.test("edge"):
                assert tuple(style_transform.children())[0].edge == "long"

            with subtests.test("interpolation_mode"):
                assert (
                    tuple(style_transform.children())[0].interpolation_mode == "bicubic"
                    if impl_params and instance_norm
                    else "bilinear"
                )


@pytest.mark.slow
def test_ulyanov_et_al_2016_images_smoke(subtests):
    for name, image in data.ulyanov_et_al_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_ulyanov_et_al_2016_images(subtests):
    for name, image in data.ulyanov_et_al_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_ulyanov_et_al_2016_dataset(subtests, mocker):
    mocker.patch(
        "pystiche.data.datasets.ImageFolderDataset._collect_image_files",
        return_value=[],
    )
    dataset = data.ulyanov_et_al_2016_dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("transform"):
        assert isinstance(
            dataset.transform, type(data.ulyanov_et_al_2016_content_transform())
        )


def test_ulyanov_et_al_2016_batch_sampler(subtests):
    data_source = ()
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            batch_sampler = data.ulyanov_et_al_2016_batch_sampler(
                data_source, impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(batch_sampler, FiniteCycleBatchSampler)

            with subtests.test("num_batches"):
                assert (
                    batch_sampler.num_batches == 200
                    if not impl_params
                    else 2000
                    if instance_norm
                    else 300
                )

            with subtests.test("num_batches"):
                assert (
                    batch_sampler.batch_size == 16
                    if not impl_params
                    else 1
                    if instance_norm
                    else 4
                )


def test_ulyanov_et_al_2016_image_loader(subtests):
    dataset = ()
    image_loader = data.ulyanov_et_al_2016_image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler,
            type(data.ulyanov_et_al_2016_batch_sampler(dataset)),
        )
