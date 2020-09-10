from os import path
from typing import Any, Dict, List, Optional, Sized, Tuple, Union, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets.utils import download_and_extract_archive

from pystiche.data import ImageFolderDataset
from pystiche.image import transforms
from pystiche.image.transforms import functional as F
from pystiche.image.utils import extract_image_size
from pystiche.misc import verify_str_arg

from ..utils import OptionalGrayscaleToFakegrayscale
from ._augmentation import augmentation

__all__ = [
    "ClampSize",
    "style_image_transform",
    "content_image_transform",
    "WikiArt",
    "style_dataset",
    "content_dataset",
    "batch_sampler",
    "image_loader",
]


class ClampSize(transforms.Transform):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/prepare_dataset.py#L49-L68
    def __init__(
        self,
        min_edge_size: int = 800,
        max_edge_size: int = 1800,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()

        if max_edge_size < min_edge_size:
            raise ValueError(
                f"max_edge_size cannot be smaller than min_edge_size: "
                f"{max_edge_size} < {min_edge_size}"
            )

        self.max_edge_size = max_edge_size
        self.min_edge_size = min_edge_size
        self.interpolation_mode = interpolation_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        short_edge_size, long_edge_size = sorted(extract_image_size(image))

        if (
            short_edge_size >= self.min_edge_size
            and long_edge_size <= self.max_edge_size
        ):
            return image

        size: Union[int, Tuple[int, int]]
        if long_edge_size > self.max_edge_size:
            size = self.max_edge_size
            edge = "long"
        else:  # short_edge_size < self.min_edge_size
            size = (
                self.min_edge_size
                if short_edge_size / self.min_edge_size > 0.25
                else (self.min_edge_size, self.min_edge_size)
            )
            edge = "short"

        return cast(
            torch.Tensor,
            F.resize(
                image, size, edge=edge, interpolation_mode=self.interpolation_mode
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["min_edge_size"] = self.min_edge_size
        dct["max_edge_size"] = self.max_edge_size
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct


def style_image_transform(edge_size: int = 768, train: bool = False) -> nn.Sequential:
    transforms_: List[nn.Module] = [
        ClampSize(),
        transforms.ValidRandomCrop((edge_size, edge_size)),
        OptionalGrayscaleToFakegrayscale(),
    ]
    if train:
        transforms_.append(augmentation(size=edge_size))
    return nn.Sequential(*transforms_)


def content_image_transform(
    impl_params: bool = True, edge_size: int = 768, train: bool = False,
) -> nn.Sequential:
    transform = style_image_transform(edge_size=edge_size, train=train)
    if not impl_params:
        return transform

    return nn.Sequential(transforms.Rescale(2.0), *transform.children())


class WikiArt(ImageFolderDataset):
    BASE_URL = "https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf/download?path=%2F&files="
    MD5_CHECKSUMS = {
        "berthe-morisot": "0f61db2107b16a86fe38caff1a2c4125",
        "edvard-munch": "8a75855098b412f574540cb01a7e2dda",
        "el-greco": "fb9ad7ee563d5c6ed3ea9498567e655c",
        "ernst-ludwig-kirchner": "35137d813ee58505266e46e24813dc42",
        "jackson-pollock": "6fbece7f00da43b9492b835fac4cc7f8",
        "monet_water-lilies-1914": "bba66d8a3fff62a4b0b714b215636d01",
        "nicholas-roerich": "1683e2b1e4af7853bdfe1b63f8db0e26",
        "pablo-picasso": "54b99819e79f522fde04ba324215814a",
        "paul-cezanne": "7c1d841f98e21f5de3ba69a3aa25b3bd",
        "sample_photographs": "173967574bf236a21044ef593286dd6f",
        "samuel-peploe": "de82d2654ee437abdd8a6b3cdf458bfd",
        "vincent-van-gogh_road-with-cypresses-1890": "6e629d6a03c65bc510bba8b619aad291",
        "wassily-kandinsky": "b506040ee2d038b8f3767125d04bde5f",
    }
    STYLES = tuple(sorted(MD5_CHECKSUMS.keys()))

    def __init__(
        self,
        root: str,
        style: str,
        transform: Optional[nn.Module] = None,
        download: bool = False,
    ) -> None:
        self.root = root = path.abspath(path.expanduser(root))
        self.style = self._verify_style(style)

        if download:
            self.download()

        super().__init__(self.sub_dir, transform=transform)
        self.root = root

    def _verify_style(self, style: str) -> str:
        return verify_str_arg(style, "style", self.STYLES)

    @property
    def sub_dir(self) -> str:
        return path.join(self.root, self.style)

    @property
    def archive(self) -> str:
        return f"{self.sub_dir}.tar.gz"

    @property
    def md5(self) -> str:
        return self.MD5_CHECKSUMS[self.style]

    @property
    def url(self) -> str:
        return f"{self.BASE_URL}{path.basename(self.archive)}"

    def download(self) -> None:
        if path.exists(self.sub_dir):
            msg = (
                f"The directory {self.sub_dir} already exists. If you want to "
                "re-download the images, delete the folder."
            )
            raise RuntimeError(msg)

        download_and_extract_archive(
            self.url, self.root, filename=self.archive, md5=self.md5
        )


def style_dataset(
    root: str,
    style: str,
    transform: Optional[nn.Module] = None,
    download: bool = False,
) -> WikiArt:
    if transform is None:
        transform = style_image_transform()
    return WikiArt(root, style, transform=transform, download=download)


# TODO: replace this with torchvision.datasets.Places365 as soon as
#  https://github.com/pytorch/vision/pull/2610 is part of a release
def content_dataset(
    root: str, impl_params: bool = True, transform: Optional[nn.Module] = None,
) -> ImageFolderDataset:
    if transform is None:
        transform = content_image_transform(impl_params=impl_params)
    return ImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized, impl_params: bool = True, num_samples: Optional[int] = None,
) -> RandomSampler:

    if num_samples is None:
        # The num_iterations are split up into multiple epochs with corresponding
        # num_batches:
        # The number of epochs is defined in _nst.training .
        # 300_000 = 1 * 300_000
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L68
        # 300_000 = 3 * 100_000
        num_samples = 300_000 if impl_params else 100_000

    return RandomSampler(data_source, replacement=True, num_samples=num_samples)


batch_sampler_ = batch_sampler


def image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if batch_sampler is None:
        batch_sampler = cast(Sampler, batch_sampler_(dataset, impl_params=impl_params))

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
