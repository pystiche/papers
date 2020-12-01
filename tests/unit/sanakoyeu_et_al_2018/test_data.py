import functools
import os
from distutils import dir_util
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.utils.data import DataLoader

import pystiche
import pystiche.image.transforms.functional as F
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.data import DownloadableImageCollection, ImageFolderDataset
from pystiche.image import (
    extract_edge_size,
    extract_image_size,
    make_single_image,
    transforms,
    write_image,
)
from pystiche.misc import to_2d_arg
from pystiche_papers.data.utils import (
    RandomNumIterationsBatchSampler,
    SequentialNumIterationsBatchSampler,
)
from pystiche_papers.utils import make_reproducible

from . import make_paper_mock_target
from tests import asserts, mocks, parametrize
from tests.utils import make_tar


@pytest.fixture()
def patch_collect_images(mocker):
    return mocker.patch(
        make_paper_mock_target("_data", "ImageFolderDataset", "_collect_image_files"),
        return_value=[],
    )


def test_ClampSize_invalid_sizes():
    with pytest.raises(ValueError):
        paper.ClampSize(min_edge_size=2, max_edge_size=1)


def test_ClampSize_normal_image(image):
    short_edge_size, long_edge_size = sorted(extract_image_size(image))
    min_edge_size = short_edge_size // 2
    max_edge_size = long_edge_size * 2
    transform = paper.ClampSize(
        min_edge_size=min_edge_size, max_edge_size=max_edge_size
    )

    assert transform(image) is image


def test_ClampSize_too_large_image(image):
    long_edge_size = max(extract_image_size(image))
    max_edge_size = long_edge_size // 2
    min_edge_size = max_edge_size - 1
    transform = paper.ClampSize(
        min_edge_size=min_edge_size, max_edge_size=max_edge_size
    )

    actual = transform(image)
    expected = F.resize(image, max_edge_size, edge="long")
    ptu.assert_allclose(actual, expected)


def test_ClampSize_too_small_image(image):
    short_edge_size = min(extract_image_size(image))
    min_edge_size = short_edge_size * 2
    max_edge_size = min_edge_size + 1
    transform = paper.ClampSize(
        min_edge_size=min_edge_size, max_edge_size=max_edge_size
    )

    actual = transform(image)
    expected = F.resize(image, min_edge_size, edge="short")
    ptu.assert_allclose(actual, expected)


def test_ClampSize_far_too_small_image(image_small_landscape):
    short_edge_size = min(extract_image_size(image_small_landscape))
    min_edge_size = short_edge_size * 5
    max_edge_size = min_edge_size + 1
    transform = paper.ClampSize(
        min_edge_size=min_edge_size, max_edge_size=max_edge_size
    )

    actual = transform(image_small_landscape)
    expected = F.resize(image_small_landscape, (min_edge_size, min_edge_size))
    ptu.assert_allclose(actual, expected)


def test_ClampSize_repr(subtests):
    kwargs = {"min_edge_size": 1, "max_edge_size": 2, "interpolation_mode": "bicubic"}
    transform = paper.ClampSize(**kwargs)

    assert isinstance(repr(transform), str)

    assert_property_in_repr = functools.partial(
        asserts.assert_property_in_repr, repr(transform)
    )
    for name, value in kwargs.items():
        with subtests.test(name):
            assert_property_in_repr(name, value)


def test_OptionalUpsample_normal_image(image):
    short_edge_size = extract_edge_size(image, edge="short")
    min_edge_size = short_edge_size // 2
    transform = paper.OptionalUpsample(min_edge_size=min_edge_size,)

    assert transform(image) is image


def test_OptionalUpsample_too_small(image):
    short_edge_size = extract_edge_size(image, edge="short")
    min_edge_size = short_edge_size * 2
    transform = paper.OptionalUpsample(min_edge_size=min_edge_size)

    actual = transform(image)
    expected = F.resize(image, min_edge_size, edge="short")
    ptu.assert_allclose(actual, expected)


def test_OptionalUpsample_repr(subtests):
    kwargs = {"min_edge_size": 1, "interpolation_mode": "bicubic"}
    transform = paper.OptionalUpsample(**kwargs)

    assert isinstance(repr(transform), str)

    assert_property_in_repr = functools.partial(
        asserts.assert_property_in_repr, repr(transform)
    )
    for name, value in kwargs.items():
        with subtests.test(name):
            assert_property_in_repr(name, value)


def test_RandomCrop_size(mocker, image_small_landscape):
    mocker.patch(
        mocks.make_mock_target("sanakoyeu_et_al_2018", "_data", "_adapted_uniform_int"),
        side_effect=lambda shape, low, high, same_on_batch: torch.ones(
            shape, dtype=torch.int32
        ).mul(high),
    )
    edge_size = extract_edge_size(image_small_landscape) // 2
    transform = paper.RandomCrop(edge_size, p=1.0)

    output_image = transform(image_small_landscape)
    assert extract_image_size(output_image) == to_2d_arg(edge_size)


def test_RandomCrop_repr(subtests):
    kwargs = dict(
        size=(1, 1), interpolation="bicubic", same_on_batch=True, align_corners=True
    )
    transform = paper.RandomCrop(**kwargs)

    assert isinstance(repr(transform), str)

    assert_property_in_repr = functools.partial(
        asserts.assert_property_in_repr, repr(transform)
    )
    for name, value in kwargs.items():
        with subtests.test(name):
            assert_property_in_repr(name, value)


def test_image_transform():
    edge_size = 768
    make_reproducible()
    image = torch.rand(1, 3, 800, 800)

    transform = paper.image_transform(impl_params=False, edge_size=edge_size)
    make_reproducible()
    actual = transform(image)

    transform = nn.Sequential(
        paper.OptionalUpsample(edge_size), transforms.ValidRandomCrop(edge_size),
    )
    make_reproducible()
    expected = transform(image)

    ptu.assert_allclose(actual, expected)


def test_image_transform_impl_params():
    edge_size = 768
    make_reproducible()
    image = torch.rand(1, 3, 800, 800)

    transform = paper.image_transform(impl_params=True, edge_size=edge_size)
    make_reproducible()
    actual = transform(image)

    transform = nn.Sequential(
        paper.ClampSize(),
        paper.pre_crop_augmentation(),
        paper.RandomCrop(edge_size, p=1.0),
        paper.post_crop_augmentation(),
    )
    make_reproducible()
    expected = transform(image)

    ptu.assert_allclose(actual, expected)


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)


def test_WikiArt_unknown_style(tmpdir):
    style = "unknown"
    with pytest.raises(ValueError):
        paper.style_dataset(tmpdir, style)


@pytest.fixture
def wiki_art_style(monkeypatch):
    style = "style"
    monkeypatch.setitem(paper.WikiArt.MD5_CHECKSUMS, style, None)
    monkeypatch.setattr(
        make_paper_mock_target("_data", "WikiArt", "STYLES"),
        (*paper.WikiArt.STYLES, style),
    )
    return style


@pytest.fixture
def wiki_art_dir(wiki_art_style, tmpdir, image_small_0, image_small_1, image_small_2):
    images = (image_small_0, image_small_1, image_small_2)

    sub_dir = path.join(tmpdir, wiki_art_style)
    os.mkdir(sub_dir)
    for idx, image in enumerate(images):
        write_image(image, path.join(sub_dir, f"{idx}.png"))
    make_tar(f"{sub_dir}.tar.gz", sub_dir)
    dir_util.remove_tree(sub_dir)

    return tmpdir, wiki_art_style, images


def test_WikiArt_extract(wiki_art_dir):
    root, style, images = wiki_art_dir

    dataset = paper.WikiArt(root, style, download=True)

    actual = iter(dataset)
    expected = (make_single_image(image) for image in images)

    assert {pystiche.TensorKey(tensor) for tensor in actual} == {
        pystiche.TensorKey(tensor) for tensor in expected
    }


def test_WikiArt_download_existing_sub_dir(wiki_art_style, tmpdir):
    os.mkdir(path.join(tmpdir, wiki_art_style))

    with pytest.raises(RuntimeError):
        paper.WikiArt(tmpdir, wiki_art_style, download=True)


def test_style_dataset_smoke(
    subtests, wiki_art_style, patch_collect_images, tmpdir, style_image
):
    root = tmpdir

    dataset = paper.style_dataset(root, wiki_art_style, download=False)
    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("root"):
        assert dataset.root == root

    with subtests.test("style"):
        assert dataset.style == wiki_art_style

    with subtests.test("transform"):
        make_reproducible()
        actual = dataset.transform(style_image)

        transform = paper.image_transform()
        make_reproducible()
        expected = transform(style_image)
        ptu.assert_allclose(actual, expected)


def test_content_dataset_transform_impl_params(
    subtests, patch_collect_images, content_image
):
    dataset = paper.content_dataset("root")

    assert isinstance(dataset, paper.Places365Subset)

    with subtests.test("transform"):
        make_reproducible()
        actual = dataset.transform(content_image)

        transform = nn.Sequential(transforms.Rescale(2.0), paper.image_transform())
        make_reproducible()
        expected = transform(content_image)
        ptu.assert_allclose(actual, expected)


@parametrize.data(
    ("impl_params", "batch_sampler_type", "num_iterations"),
    (
        (True, RandomNumIterationsBatchSampler, 300_000),
        (False, SequentialNumIterationsBatchSampler, 100_000),
    ),
)
def test_batch_sampler(impl_params, batch_sampler_type, num_iterations):
    batch_sampler = paper.batch_sampler((), impl_params=impl_params)

    assert isinstance(batch_sampler, batch_sampler_type)
    assert batch_sampler.num_iterations == num_iterations


def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler, type(paper.batch_sampler(dataset)),
        )
