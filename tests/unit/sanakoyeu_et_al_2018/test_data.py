import pytest

import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

import pystiche.image.transforms.functional as F
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.data import ImageFolderDataset
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.utils import make_reproducible

from tests.asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_image_transform():
    make_reproducible()
    image = torch.rand(1, 1, 16, 31)

    content_transform = paper.content_transform(impl_params=False)
    actual = content_transform(image)

    desired = F.grayscale_to_fakegrayscale(F.resize(image[:, :, :16, :16], 256))

    ptu.assert_allclose(actual, desired)


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
        "pystiche_papers.sanakoyeu_et_al_2018._data.ImageFolderDataset._collect_image_files",
        return_value=[],
    )
    dataset = paper.dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("transform"):
        assert isinstance(dataset.transform, type(paper.image_transform()))


def test_batch_sampler(subtests):
    batch_sampler = paper.batch_sampler(())

    assert isinstance(batch_sampler, FiniteCycleBatchSampler)

    with subtests.test("num_batches"):
        assert batch_sampler.batch_size == 1


def test_batch_sampler_num_batches_default(subtests):
    for impl_params, num_batches in ((True, 300_000), (False, 100_000)):
        with subtests.test(impl_params=impl_params):
            batch_sampler = paper.batch_sampler((), impl_params=impl_params)
            assert batch_sampler.num_batches == num_batches



def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler, type(paper.batch_sampler(dataset)),
        )
