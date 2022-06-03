import pytorch_testing_utils as ptu
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche.data import DownloadableImageCollection, ImageFolderDataset
from pystiche_papers import utils
from pystiche_papers.data.utils import FiniteCycleBatchSampler

from .utils import impl_params_and_instance_norm


@impl_params_and_instance_norm
def test_content_transform(subtests, content_image, impl_params, instance_norm):
    edge_size = 16

    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )
    hyper_parameters.content_transform.edge_size = edge_size

    content_transform = paper.content_transform(
        impl_params=impl_params,
        instance_norm=instance_norm,
        hyper_parameters=hyper_parameters,
    )

    utils.make_reproducible()
    actual = content_transform(content_image)

    if impl_params:
        if instance_norm:
            # Since the transform involves an uncontrollable random component, we can't
            # do a reference test here
            return

        desired = F.resize(content_image, edge_size)
    else:
        transform = transforms.CenterCrop(edge_size)
        desired = transform(content_image)

    ptu.assert_allclose(actual, desired)


@impl_params_and_instance_norm
def test_content_transform_grayscale_image(
    subtests, content_image, impl_params, instance_norm
):
    content_image = F.rgb_to_grayscale(content_image)
    edge_size = 16

    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )
    hyper_parameters.content_transform.edge_size = edge_size

    content_transform = paper.content_transform(
        impl_params=impl_params,
        instance_norm=instance_norm,
        hyper_parameters=hyper_parameters,
    )
    if instance_norm:
        utils.make_reproducible()
    actual = content_transform(content_image)

    if impl_params:
        if instance_norm:
            # Since the transform involves an uncontrollable random component, we can't
            # do a reference test here
            return

        transform_image = F.resize(content_image, edge_size)
    else:
        transform = transforms.CenterCrop(edge_size)
        transform_image = transform(content_image)

    desired = transform_image.repeat(1, 3, 1, 1)

    ptu.assert_allclose(actual, desired)


@impl_params_and_instance_norm
def test_style_transform(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).style_transform

    style_transform = paper.style_transform(
        impl_params=impl_params, instance_norm=instance_norm
    )

    with subtests.test("edge_size"):
        assert style_transform.edge_size == hyper_parameters.edge_size

    with subtests.test("interpolation"):
        assert style_transform.interpolation == hyper_parameters.interpolation


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


@impl_params_and_instance_norm
def test_batch_sampler(subtests, impl_params, instance_norm):
    data_source = ()
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).batch_sampler

    batch_sampler = paper.batch_sampler(
        data_source, impl_params=impl_params, instance_norm=instance_norm
    )

    assert isinstance(batch_sampler, FiniteCycleBatchSampler)

    with subtests.test("num_batches"):
        assert batch_sampler.num_batches == hyper_parameters.num_batches

    with subtests.test("num_size"):
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
