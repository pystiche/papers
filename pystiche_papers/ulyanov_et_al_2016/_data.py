from typing import cast, List, Optional, Sized, Tuple, Union
from urllib.parse import urljoin

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import functional as F

from pystiche.data import (
    CreativeCommonsLicense,
    DownloadableImage,
    DownloadableImageCollection,
    ExpiredCopyrightLicense,
    ImageFolderDataset,
)
from pystiche.image import extract_image_size
from pystiche_papers.data.utils import FiniteCycleBatchSampler
from pystiche_papers.utils import HyperParameters

from ..utils import OptionalGrayscaleToFakegrayscale
from ._utils import hyper_parameters as _hyper_parameters

__all__ = [
    "content_transform",
    "style_transform",
    "images",
    "dataset",
    "batch_sampler",
    "image_loader",
]


class ValidRandomCrop(nn.Module):
    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size

    @staticmethod
    def get_params(
        image_size: Tuple[int, int], crop_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        image_height, image_width = image_size
        crop_height, crop_width = crop_size

        def randint(range: int) -> int:
            if range < 0:
                raise RuntimeError(
                    "The crop size has to be smaller or equal to the image size."
                )
            return int(torch.randint(range + 1, (), dtype=torch.long))

        top = randint(image_height - crop_height)
        left = randint(image_width - crop_width)
        return top, left

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        top, left = self.get_params(extract_image_size(image), self.size)
        height, width = self.size
        return cast(
            torch.Tensor,
            F.crop(
                image,
                top=top,
                left=left,
                height=height,
                width=width,
            ),
        )


def content_transform(
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Sequential:
    r"""Content transform from :cite:`ULVL2016,UVL2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )
    edge_size = hyper_parameters.content_transform.edge_size

    transforms_: List[nn.Module] = []
    if impl_params:
        if instance_norm:
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/datasets/style.lua#L83
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/datasets/transforms.lua#L62-L92
            transforms_.append(ValidRandomCrop(edge_size))
        else:
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_process.lua#L30
            # https://github.com/torch/image/blob/master/doc/simpletransform.md#res-imagescalesrc-width-height-mode
            transforms_.append(
                transforms.Resize(
                    (edge_size, edge_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
    else:
        transforms_.append(transforms.CenterCrop(edge_size))

    transforms_.append(OptionalGrayscaleToFakegrayscale())
    return nn.Sequential(*transforms_)


# TODO: refactor to a common transform
class LongEdgeResize(nn.Module):
    def __init__(
        self,
        edge_size: int,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.edge_size = edge_size
        self.interpolation = interpolation

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        old_height, old_width = extract_image_size(image)
        if old_height > old_width:
            new_height = self.edge_size
            new_width = int(new_height / old_height * old_width)
        else:
            new_width = self.edge_size
            new_height = int(new_width / old_width * old_height)

        return cast(
            torch.Tensor,
            F.resize(image, [new_height, new_width], interpolation=self.interpolation),
        )


def style_transform(
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> LongEdgeResize:
    r"""Style transform from :cite:`ULVL2016,UVL2017`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    # https://github.com/torch/image/blob/master/doc/simpletransform.md#res-imagescalesrc-size-mode
    return LongEdgeResize(
        hyper_parameters.style_transform.edge_size,
        interpolation=hyper_parameters.style_transform.interpolation,
    )


def images() -> DownloadableImageCollection:
    r"""Images from :cite:`ULVL2016,UVL2017`."""
    base_ulyanov = (
        "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/master/data/"
    )
    base_ulyanov_suppl = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary/"
    readme_ulyanov = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/data/readme_pics/"
    content_base_ulyanov = urljoin(base_ulyanov, "readme_pics/")
    content_images = {
        "karya": DownloadableImage(
            urljoin(content_base_ulyanov, "karya.jpg"),
            md5="232b2f03a5d20c453a41a0e6320f27be",
        ),
        "tiger": DownloadableImage(
            urljoin(content_base_ulyanov, "tiger.jpg"),
            md5="e82bf374da425fb2c2e2a35a5a751989",
        ),
        "neckarfront": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg",
            title="Tübingen Neckarfront",
            author="Andreas Praefcke",
            license=CreativeCommonsLicense(("by",), version="3.0"),
            md5="dc9ad203263f34352e18bc29b03e1066",
            file="tuebingen_neckarfront__andreas_praefcke.jpg",
        ),
        "tower_of_babel": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/f/fc/Pieter_Bruegel_the_Elder_-_The_Tower_of_Babel_%28Vienna%29_-_Google_Art_Project_-_edited.jpg",
            title="The Tower of Babel",
            author="Pieter Bruegel",
            date="1563",
            license=ExpiredCopyrightLicense(1563),
            md5="1e113716c8aad6c2ca826ae0b83ffc76",
            file="the_tower_of_babel.jpg",
        ),
        "bird": DownloadableImage(
            urljoin(base_ulyanov_suppl, "bird.jpg"),
            md5="74dde9fad4749e7ff3cd4eca6cb43d0d",
        ),
        "kitty": DownloadableImage(
            urljoin(readme_ulyanov, "kitty.jpg"),
            md5="98262bd8f5ae25f8329158d2c2c66ad0",
        ),
    }

    base_johnson = (
        "https://raw.githubusercontent.com/jcjohnson/fast-neural-style/master/images/"
    )
    style_base_johnson = urljoin(base_johnson, "styles/")

    base_ulyanov_suppl_style = "https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/texture_nets_v1/supplementary//stylization_models/"
    style_images = {
        "candy": DownloadableImage(
            urljoin(style_base_johnson, "candy.jpg"),
            md5="00a0e3aa9775546f98abf6417e3cb478",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_base_johnson, "the_scream.jpg"),
            md5="619b4f42c84d2b62d3518fb20fa619c2",
        ),
        "jean_metzinger": DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/c/c9/Robert_Delaunay%2C_1906%2C_Portrait_de_Metzinger%2C_oil_on_canvas%2C_55_x_43_cm%2C_DSC08255.jpg",
            title="Portrait of Jean Metzinger",
            author="Jean Metzinger",
            date="1906",
            license=ExpiredCopyrightLicense(1906),
            md5="3539d50d2808b8eec5b05f892d8cf1e1",
            file="jean_metzinger.jpg",
        ),
        "mosaic": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "mosaic.jpg"),
            md5="4f05f1e12961cebf41bd372d909342b3",
        ),
        "pleades": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "pleades.jpg"),
            md5="6fc41ac30c2852a5454a0ead2f479dc9",
        ),
        "starry": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "starry.jpg"),
            md5="c6d94f7962466b2e80a64ae82523242a",
        ),
        "turner": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "turner.jpg"),
            md5="7fdd9603a5182dcef23d7fb1c5217888",
        ),
    }
    return DownloadableImageCollection({**content_images, **style_images})


def dataset(
    root: str,
    impl_params: bool = True,
    instance_norm: bool = True,
    transform: Optional[nn.Module] = None,
) -> ImageFolderDataset:
    if transform is None:
        transform = content_transform(
            impl_params=impl_params, instance_norm=instance_norm
        )
    return ImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized,
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> FiniteCycleBatchSampler:
    r"""Batch sampler from :cite:`ULVL2016,UVL2017`.

    Args:
        data_source: Dataset to sample from.
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        instance_norm: Switch the behavior and hyper-parameters between both
            publications of the original authors. For details see
            :ref:`here <ulyanov_et_al_2016-instance_norm>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.ulyanov_et_al_2016.hyper_parameters` is used.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    return FiniteCycleBatchSampler(
        data_source,
        num_batches=hyper_parameters.batch_sampler.num_batches,
        batch_size=hyper_parameters.batch_sampler.batch_size,
    )


batch_sampler_ = batch_sampler


def image_loader(
    dataset: Sized,
    impl_params: bool = True,
    instance_norm: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if batch_sampler is None:
        batch_sampler = batch_sampler_(
            dataset, impl_params=impl_params, instance_norm=instance_norm
        )

    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
