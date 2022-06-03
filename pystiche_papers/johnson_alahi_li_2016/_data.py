from typing import cast, List, Optional, Sized
from urllib.parse import urljoin

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from pystiche import image
from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    ImageFolderDataset,
)
from pystiche.image import extract_image_size
from pystiche_papers.utils import HyperParameters

from ..data.utils import FiniteCycleBatchSampler
from ..utils.transforms import OptionalGrayscaleToFakegrayscale
from ._utils import hyper_parameters as _hyper_parameters, preprocessor as _preprocessor

__all__ = [
    "content_transform",
    "style_transform",
    "images",
    "dataset",
    "batch_sampler",
    "image_loader",
]


class TopLeftCropToMultiple(nn.Module):
    def __init__(self, multiple: int = 16):
        super().__init__()
        self.multiple = multiple

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        old_height, old_width = image.extract_image_size(input_image)
        new_height = old_height - old_height % self.multiple
        new_width = old_width - old_width % self.multiple

        return input_image[..., :new_height, :new_width]


def content_transform(
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    r"""Content image transformation from :cite:`JAL2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <johnson_alahi_li_2016-impl_params>`.

            Additionally, if ``True``, appends the
            :func:`~pystiche_papers.johnson_alahi_li_2016.preprocessor` as a last
            transformation step.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    transforms_: List[nn.Module] = [
        TopLeftCropToMultiple(),
        transforms.Resize(hyper_parameters.content_transform.image_size),
        OptionalGrayscaleToFakegrayscale(),
    ]
    if impl_params:
        transforms_.append(_preprocessor())

    return nn.Sequential(*transforms_)


class LongEdgeResize(nn.Module):
    def __init__(self, edge_size: int) -> None:
        super().__init__()
        self.edge_size = edge_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        old_height, old_width = extract_image_size(image)
        if old_height > old_width:
            new_height = self.edge_size
            new_width = int(new_height / old_height * old_width)
        else:
            new_width = self.edge_size
            new_height = int(new_width / old_width * old_height)

        return cast(torch.Tensor, F.resize(image, [new_height, new_width]))


def style_transform(
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Module:
    r"""Style image transformation from :cite:`JAL2016`.

    Args:
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return LongEdgeResize(hyper_parameters.style_transform.edge_size)


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
    return DownloadableImageCollection({**content_images, **style_images})


def dataset(
    root: str,
    impl_params: bool = True,
    transform: Optional[nn.Module] = None,
) -> ImageFolderDataset:
    if transform is None:
        transform = content_transform(impl_params=impl_params)

    return ImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized,
    hyper_parameters: Optional[HyperParameters] = None,
) -> FiniteCycleBatchSampler:
    r"""Batch sampler from :cite:`JAL2016`.

    Args:
        data_source: Dataset to sample from.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.johnson_alahi_li_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return FiniteCycleBatchSampler(
        data_source,
        num_batches=hyper_parameters.batch_sampler.num_batches,
        batch_size=hyper_parameters.batch_sampler.batch_size,
    )


def image_loader(
    dataset: Sized,
    hyper_parameters: Optional[HyperParameters] = None,
    pin_memory: bool = True,
) -> DataLoader:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_sampler=batch_sampler(dataset),
        num_workers=hyper_parameters.batch_sampler.batch_size,
        pin_memory=pin_memory,
    )
