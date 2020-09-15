import functools
import os
from distutils import dir_util
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

import pystiche
import pystiche.image.transforms.functional as F
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.data import ImageFolderDataset
from pystiche.image import (
    extract_image_size,
    make_single_image,
    transforms,
    write_image,
)
from pystiche_papers.utils import make_reproducible

from . import make_paper_mock_target
from tests import asserts
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


def test_style_image_transform():
    image_size = (16, 16)
    make_reproducible()
    image = torch.rand(1, 1, 800, 800)

    make_reproducible()
    image_transform = paper.style_image_transform(image_size=image_size)
    actual = image_transform(image)

    transform = transforms.ValidRandomCrop(image_size)
    make_reproducible()
    expected = F.grayscale_to_fakegrayscale(transform(image))

    ptu.assert_allclose(actual, expected)


def test_style_image_transform_augmentation():
    image_size = (16, 16)
    make_reproducible()
    image = torch.rand(1, 1, 800, 800)

    make_reproducible()
    image_transform = paper.style_image_transform(image_size=image_size, train=True)
    actual = image_transform(image)

    make_reproducible()
    transform = nn.Sequential(
        paper.style_image_transform(image_size=image_size, train=False),
        paper.augmentation(image_size=image_size),
    )
    expected = transform(image)

    ptu.assert_allclose(actual, expected)


def test_content_image_transform_no_impl_params(subtests, content_image):
    style_transform = paper.style_image_transform()

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            content_transform = paper.content_image_transform(impl_params=impl_params)
            make_reproducible()
            actual = content_transform(content_image)

            make_reproducible()
            expected = style_transform(
                F.rescale(content_image, 2.0) if impl_params else content_image
            )

            ptu.assert_allclose(actual, expected)


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


def test_style_dataset_smoke(subtests, wiki_art_style, patch_collect_images, tmpdir):
    root = tmpdir

    dataset = paper.style_dataset(root, wiki_art_style, download=False)
    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("root"):
        assert dataset.root == root

    with subtests.test("style"):
        assert dataset.style == wiki_art_style

    with subtests.test("transform"):
        assert isinstance(dataset.transform, type(paper.style_image_transform()))


def test_content_dataset(subtests, patch_collect_images):
    dataset = paper.content_dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("transform"):
        assert isinstance(dataset.transform, type(paper.content_image_transform()))


def test_batch_sampler(subtests):
    for impl_params, num_samples in ((True, 300_000), (False, 100_000)):
        batch_sampler = paper.batch_sampler((), impl_params=impl_params)

        assert isinstance(batch_sampler, RandomSampler)

        with subtests.test(impl_params=impl_params):
            assert batch_sampler.num_samples == num_samples


def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler, type(paper.batch_sampler(dataset)),
        )
