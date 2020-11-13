from typing import Optional, Sized, Tuple
from urllib.parse import urljoin

import torch
from torch.utils.data import DataLoader, Dataset

import pystiche.image.transforms.functional as F
from pystiche import image
from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    ImageFolderDataset,
)
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

from ..data.utils import FiniteCycleBatchSampler
from ..utils.transforms import OptionalGrayscaleToFakegrayscale
from ._utils import hyper_parameters as _hyper_parameters

__all__ = [
    "content_transform",
    "style_transform",
    "images",
    "dataset",
    "batch_sampler",
    "image_loader",
]


class TopLeftCropToMultiple(transforms.Transform):
    def __init__(self, multiple: int = 16):
        super().__init__()
        self.multiple = multiple

    def calculate_size(self, input_image: torch.Tensor) -> Tuple[int, int]:
        old_height, old_width = image.extract_image_size(input_image)
        new_height = old_height - old_height % self.multiple
        new_width = old_width - old_width % self.multiple
        return new_height, new_width

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        size = self.calculate_size(input_image)
        return F.top_left_crop(input_image, size)


def content_transform(
    impl_params: bool = True, hyper_parameters: Optional[HyperParameters] = None,
) -> transforms.ComposedTransform:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    edge_size = hyper_parameters.content_transform.edge_size
    transforms_ = [
        TopLeftCropToMultiple(),
        transforms.Resize((edge_size, edge_size)),
        OptionalGrayscaleToFakegrayscale(),
    ]
    if impl_params:
        transforms_.append(transforms.CaffePreprocessing())

    return transforms.ComposedTransform(*transforms_)


def style_transform(
    hyper_parameters: Optional[HyperParameters] = None,
) -> transforms.Resize:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return transforms.Resize(
        hyper_parameters.style_transform.edge_size,
        edge=hyper_parameters.style_transform.edge,
    )


def images() -> DownloadableImageCollection:
    root = (
        "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/"
    )

    content_root = urljoin(root, "content/")
    content_images = {
        "chicago": DownloadableImage(
            urljoin(content_root, "chicago.jpg"), md5="16ea186230a8a5131b224ddde01d0dd5"
        ),
        "hoovertowernight": DownloadableImage(
            urljoin(content_root, "hoovertowernight.jpg"),
            md5="97f7bab04e1f4c852fd2499356163b15",
        ),
    }

    style_root = urljoin(root, "styles/")
    style_images = {
        "candy": DownloadableImage(
            urljoin(style_root, "candy.jpg"), md5="00a0e3aa9775546f98abf6417e3cb478"
        ),
        "composition_vii": DownloadableImage(
            urljoin(style_root, "composition_vii.jpg"),
            md5="8d4f97cb0e8b1b07dee923599ee86cbd",
        ),
        "feathers": DownloadableImage(
            urljoin(style_root, "feathers.jpg"), md5="461c8a1704b59af1cf686883b16feec6"
        ),
        "la_muse": DownloadableImage(
            urljoin(style_root, "la_muse.jpg"), md5="77262ef6985cc427f84d78784ab5c1d8"
        ),
        "mosaic": DownloadableImage(
            urljoin(style_root, "mosaic.jpg"), md5="67b11e9cb1a69df08d70d9c2c7778fba"
        ),
        "starry_night": DownloadableImage(
            urljoin(style_root, "starry_night.jpg"),
            md5="ff217acb6db32785b8651a0e316aeab3",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_root, "the_scream.jpg"),
            md5="619b4f42c84d2b62d3518fb20fa619c2",
        ),
        "udnie": DownloadableImage(
            urljoin(style_root, "udnie.jpg"), md5="6f3fa51706b21580a4b77f232d3b8ba9"
        ),
        "the_wave": DownloadableImage(
            urljoin(style_root, "wave.jpg"),
            md5="b06acee16641a2a04fb87bade8cee529",
            file="the_wave.jpg",
        ),
    }
    return DownloadableImageCollection({**content_images, **style_images},)


def dataset(
    root: str,
    impl_params: bool = True,
    transform: Optional[transforms.Transform] = None,
) -> ImageFolderDataset:
    if transform is None:
        transform = content_transform(impl_params=impl_params)

    return ImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized, hyper_parameters: Optional[HyperParameters] = None,
) -> FiniteCycleBatchSampler:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return FiniteCycleBatchSampler(
        data_source,
        num_batches=hyper_parameters.batch_sampler.num_batches,
        batch_size=hyper_parameters.batch_sampler.batch_size,
    )


def image_loader(
    dataset: Dataset,
    hyper_parameters: Optional[HyperParameters] = None,
    pin_memory: bool = True,
) -> DataLoader:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler(dataset),
        num_workers=hyper_parameters.batch_sampler.batch_size,
        pin_memory=pin_memory,
    )
