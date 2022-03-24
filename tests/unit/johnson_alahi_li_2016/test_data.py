import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader

import pystiche.image.transforms.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche.data import DownloadableImageCollection, ImageFolderDataset
from pystiche.image import transforms
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.utils import make_reproducible


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


def test_style_transform(subtests):
    style_transform = paper.style_transform()
    assert isinstance(style_transform, transforms.Resize)

    hyper_parameters = paper.hyper_parameters().style_transform

    with subtests.test("size"):
        assert style_transform.size == hyper_parameters.edge_size

    with subtests.test("edge"):
        assert style_transform.edge == hyper_parameters.edge


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)


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

    hyper_parameters = paper.hyper_parameters().batch_sampler

    with subtests.test("num_batches"):
        assert batch_sampler.num_batches == hyper_parameters.num_batches

    with subtests.test("batch_size"):
        assert batch_sampler.batch_size == hyper_parameters.batch_size


def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler,
            type(paper.batch_sampler(dataset)),
        )
