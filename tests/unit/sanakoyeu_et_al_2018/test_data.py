import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

import pystiche.image.transforms.functional as F
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.data import ImageFolderDataset
from pystiche.image import transforms
from pystiche_papers import utils
from pystiche_papers.data.utils import FiniteCycleBatchSampler


def test_image_transform():
    edge_size = 16
    utils.make_reproducible()
    image = torch.rand(1, 1, 800, 800)

    utils.make_reproducible()
    image_transform = paper.image_transform(edge_size=edge_size)
    actual = image_transform(image)

    transform = transforms.ValidRandomCrop(edge_size)
    utils.make_reproducible()
    desired = F.grayscale_to_fakegrayscale(transform(image))

    ptu.assert_allclose(actual, desired)


def test_image_transform_optional_rescale(subtests):

    with subtests.test("too large image"):
        edge_size = 16
        make_reproducible()
        image = torch.rand(1, 1, 2000, 800)

        utils.make_reproducible()
        image_transform = paper.image_transform(edge_size=edge_size)
        actual = image_transform(image)

        image = F.rescale(image, factor=0.9)
        transform = transforms.ValidRandomCrop(edge_size)
        utils.make_reproducible()
        desired = F.grayscale_to_fakegrayscale(transform(image))

        ptu.assert_allclose(actual, desired)

    with subtests.test("too small image"):
        edge_size = 16
        make_reproducible()
        image = torch.rand(1, 1, 400, 800)

        utils.make_reproducible()
        image_transform = paper.image_transform(edge_size=edge_size)
        actual = image_transform(image)
        image = F.rescale(image, factor=2)
        transform = transforms.ValidRandomCrop(edge_size)
        utils.make_reproducible()
        desired = F.grayscale_to_fakegrayscale(transform(image))

        ptu.assert_allclose(actual, desired)

    with subtests.test("very small image"):
        edge_size = 16
        make_reproducible()
        image = torch.rand(1, 1, 16, 16)

        utils.make_reproducible()
        image_transform = paper.image_transform(edge_size=edge_size)
        actual = image_transform(image)

        image = F.resize(image, (800, 800), "bilinear")
        transform = transforms.ValidRandomCrop(edge_size)
        utils.make_reproducible()
        desired = F.grayscale_to_fakegrayscale(transform(image))

        ptu.assert_allclose(actual, desired)


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
