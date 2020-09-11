import itertools

import pytorch_testing_utils as ptu
from torch.utils.data import DataLoader

import pystiche.image.transforms.functional as F
import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche.data import DownloadableImageCollection, ImageFolderDataset
from pystiche.image import transforms
from pystiche_papers import utils
from pystiche_papers.data.utils import FiniteCycleBatchSampler


def test_content_transform(subtests, content_image):
    edge_size = 16

    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            content_transform = paper.content_transform(
                edge_size=edge_size,
                impl_params=impl_params,
                instance_norm=instance_norm,
            )
            if instance_norm:
                utils.make_reproducible()
            actual = content_transform(content_image)

            if impl_params:
                if instance_norm:
                    transform = transforms.ValidRandomCrop(edge_size)
                    utils.make_reproducible()
                    desired = transform(content_image)
                else:
                    desired = F.resize(content_image, edge_size)
            else:
                transform = transforms.CenterCrop(edge_size)
                desired = transform(content_image)

            ptu.assert_allclose(actual, desired)


def test_content_transform_grayscale_image(subtests, content_image):
    content_image = F.rgb_to_grayscale(content_image)
    edge_size = 16

    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            content_transform = paper.content_transform(
                edge_size=edge_size,
                impl_params=impl_params,
                instance_norm=instance_norm,
            )
            if instance_norm:
                utils.make_reproducible()
            actual = content_transform(content_image)

            if impl_params:
                if instance_norm:
                    transform = transforms.ValidRandomCrop(edge_size)
                    utils.make_reproducible()
                    transform_image = transform(content_image)
                else:
                    transform_image = F.resize(content_image, edge_size)
            else:
                transform = transforms.CenterCrop(edge_size)
                transform_image = transform(content_image)

            desired = F.grayscale_to_fakegrayscale(transform_image)

            ptu.assert_allclose(actual, desired)


def test_style_transform(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            style_transform = paper.style_transform(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(style_transform, transforms.Resize)

            with subtests.test("edge_size"):
                assert style_transform.size == 256

            with subtests.test("edge"):
                assert style_transform.edge == "long"

            with subtests.test("interpolation_mode"):
                assert (
                    style_transform.interpolation_mode == "bicubic"
                    if impl_params and instance_norm
                    else "bilinear"
                )


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)


def test_dataset(subtests, mocker):
    mocker.patch(
        "pystiche_papers.ulyanov_et_al_2016._data.ImageFolderDataset._collect_image_files",
        return_value=[],
    )
    dataset = paper.dataset("root")

    assert isinstance(dataset, ImageFolderDataset)

    with subtests.test("content_transform"):
        assert isinstance(dataset.transform, type(paper.content_transform()))


def test_batch_sampler(subtests):
    data_source = ()
    configs = (
        (True, True, 2000, 1),
        (True, False, 300, 4),
        (False, True, 200, 16),
        (False, False, 200, 16),
    )
    for impl_params, instance_norm, num_batches, batch_size in configs:
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            batch_sampler = paper.batch_sampler(
                data_source, impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(batch_sampler, FiniteCycleBatchSampler)

            with subtests.test("num_batches"):
                assert batch_sampler.num_batches == num_batches

            with subtests.test("num_size"):
                assert batch_sampler.batch_size == batch_size


def test_image_loader(subtests):
    dataset = ()
    image_loader = paper.image_loader(dataset)

    assert isinstance(image_loader, DataLoader)

    with subtests.test("batch_sampler"):
        assert isinstance(
            image_loader.batch_sampler, type(paper.batch_sampler(dataset)),
        )
