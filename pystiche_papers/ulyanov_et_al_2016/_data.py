from typing import List, Iterable, Sized, Optional, Tuple, cast, Union, Iterator
from urllib.parse import urljoin

import math
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, IterableDataset

from pystiche.data import (
    CreativeCommonsLicense,
    DownloadableImage,
    DownloadableImageCollection,
    ExpiredCopyrightLicense,
    ImageFolderDataset,
)
from pystiche.image import extract_edge_size, extract_image_size
from pystiche_papers.utils import HyperParameters

from ..utils import OptionalGrayscaleToFakegrayscale
from ._utils import hyper_parameters as _hyper_parameters

__all__ = [
    "content_transform",
    "style_transform",
    "stylization_transform",
    "images",
    "dataset",
    "image_loader",
]

LICENSE_ULYANOV = (
    "The image is part of a repository that is published under the Apache "
    "License"
    "(https://github.com/DmitryUlyanov/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/LICENSE#L1)."
    "Proceed at your own risk."
)

LICENSE_JOHNSON = (
    "The image is part of a repository that is published for personal and "
    "research use only "
    "(https://github.com/jcjohnson/fast-neural-style/blob/master/README.md#license)."
    "Proceed at your own risk."
)


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
) -> nn.Module:
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


class OptionalResizeCenterCropToMultiple(nn.Module):
    def __init__(self, multiple: int = 64):
        super().__init__()
        self.multiple = multiple

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        old_height, old_width = input_image.shape[-2:]
        if old_height % self.multiple == 0 and old_width % self.multiple == 0:
            return input_image

        min_length = min([old_height, old_width])
        new_length = math.ceil(min_length / self.multiple) * self.multiple

        output_image = F.resize(input_image, new_length)
        output_image = F.center_crop(output_image, new_length)
        return cast(torch.Tensor, output_image)


def stylization_transform(
    impl_params: bool = True,
    instance_norm: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> nn.Module:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )

    if impl_params:
        # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/test.lua#L37
        # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/stylization_process.lua#L30
        edge_size = hyper_parameters.content_transform.edge_size
        return cast(nn.Module, transforms.Resize((edge_size, edge_size)))
    else:
        # No image pre-processing is described in the paper. However, images with a
        # height or width that is not divisible by 64 will result in a RuntimeError.
        # In order to be able to use the stylisation for all images, a resize and
        # cropping is carried out here.
        return OptionalResizeCenterCropToMultiple(multiple=64)


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
            license=LICENSE_ULYANOV,
            md5="232b2f03a5d20c453a41a0e6320f27be",
        ),
        "tiger": DownloadableImage(
            urljoin(content_base_ulyanov, "tiger.jpg"),
            license=LICENSE_ULYANOV,
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
            license=LICENSE_ULYANOV,
            md5="74dde9fad4749e7ff3cd4eca6cb43d0d",
        ),
        "kitty": DownloadableImage(
            urljoin(readme_ulyanov, "kitty.jpg"),
            license=LICENSE_ULYANOV,
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
            license=LICENSE_JOHNSON,
            md5="00a0e3aa9775546f98abf6417e3cb478",
        ),
        "the_scream": DownloadableImage(
            urljoin(style_base_johnson, "the_scream.jpg"),
            license=LICENSE_JOHNSON,
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
            license=LICENSE_ULYANOV,
            md5="4f05f1e12961cebf41bd372d909342b3",
        ),
        "pleades": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "pleades.jpg"),
            license=LICENSE_ULYANOV,
            md5="6fc41ac30c2852a5454a0ead2f479dc9",
        ),
        "starry": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "starry.jpg"),
            license=LICENSE_ULYANOV,
            md5="c6d94f7962466b2e80a64ae82523242a",
        ),
        "turner": DownloadableImage(
            urljoin(base_ulyanov_suppl_style, "turner.jpg"),
            license=LICENSE_ULYANOV,
            md5="7fdd9603a5182dcef23d7fb1c5217888",
        ),
    }
    return DownloadableImageCollection({**content_images, **style_images})


class Dataset(IterableDataset):
    def __init__(
        self,
        dataset: Sized,
        *,
        min_size: int,
        num_samples: int,
        transform: nn.Module,
    ):
        self.dataset = dataset
        self.min_size = min_size
        self.num_samples = num_samples
        self.transform = transform

        # Like itertools.cycle but without caching
        def cycle(iterable: Iterable) -> Iterator:
            while True:
                for item in iterable:
                    yield item

        self.data_samples = iter(cycle(cast(Iterable, self.dataset)))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator:
        num_samples = 0
        while num_samples < self.num_samples:
            sample = next(self.data_samples)
            if extract_edge_size(sample, edge="short") >= self.min_size:
                # Images that are too small are skipped by the original DataLoader and
                # the next image is used.
                # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/dataloader.lua#L91-L100
                yield self.transform(sample)
                num_samples += 1


def dataset(
    root: str,
    impl_params: bool = True,
    instance_norm: bool = True,
    transform: Optional[nn.Module] = None,
    hyper_parameters: Optional[HyperParameters] = None,
) -> Sized:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )
    if transform is None:
        transform = content_transform(
            impl_params=impl_params,
            instance_norm=instance_norm,
            hyper_parameters=hyper_parameters,
        )
    return Dataset(
        ImageFolderDataset(root),
        min_size=hyper_parameters.content_transform.edge_size,
        num_samples=hyper_parameters.num_batches * hyper_parameters.batch_size,
        transform=transform,
    )


def image_loader(
    dataset: Sized,
    impl_params: bool = True,
    instance_norm: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
) -> DataLoader:
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(
            impl_params=impl_params, instance_norm=instance_norm
        )
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=hyper_parameters.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
