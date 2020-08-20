from typing import Any, Dict, Optional, Sized, cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pystiche.data import ImageFolderDataset
from pystiche.image.transforms import ComposedTransform, Transform, ValidRandomCrop
from pystiche.image.transforms import functional as F
from pystiche.image.utils import extract_image_size

from ..data.utils import FiniteCycleBatchSampler
from ..utils import OptionalGrayscaleToFakegrayscale


def image_transform(edge_size: int = 768,) -> ComposedTransform:
    class OptionalRescale(Transform):
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/prepare_dataset.py#L49-L68
        def __init__(
            self, interpolation_mode: str = "bilinear",
        ):
            super().__init__()
            self.interpolation_mode = interpolation_mode

        def forward(
            self, image: torch.Tensor, maximal_size: int = 1800, minimal_size: int = 800
        ) -> torch.Tensor:
            if max(image.shape) > maximal_size:
                return F.rescale(
                    image, factor=maximal_size / max(extract_image_size(image))
                )
            if min(image.shape) < minimal_size:
                alpha = minimal_size / float(min(extract_image_size(image)))
                if alpha < 4.0:
                    return F.rescale(image, factor=alpha)
                else:
                    return cast(
                        torch.Tensor,
                        F.resize(
                            image, (minimal_size, minimal_size), self.interpolation_mode
                        ),
                    )
            return image

        def _properties(self) -> Dict[str, Any]:
            dct = super()._properties()
            if self.interpolation_mode != "bilinear":
                dct["interpolation_mode"] = self.interpolation_mode
            return dct

    transforms = (
        OptionalRescale(),
        ValidRandomCrop((edge_size, edge_size)),
        OptionalGrayscaleToFakegrayscale(),
    )
    return ComposedTransform(*transforms)


# TODO: Implement this
# def images(root: Optional[str] = None,
#   download: bool = True,
#   overwrite: bool = False):
#
#     # base_sanakoyeu =
#     "https://hcicloud.iwr.uni-heidelberg.de/index.php/s/NcJj2oLBTYuT1tf/download?path=%2F&files="
#     # tar1 = "berthe-morisot.tar.gz"
#     # tar2 = "edvard-munch.tar.gz"
#     # tar3 = "el-greco.tar.gz"
#     # tar4 = "ernst-ludwig-kirchner.tar.gz"
#     # tar5 = "jackson-pollock.tar.gz"
#     # tar6 = "monet_water-lilies-1914.tar.gz"
#     # tar7 = "nicholas-roerich.tar.gz"
#     # tar8 = "pablo-picasso.tar.gz"
#     # tar9 = "paul-cezanne.tar.gz"
#     # tar10 = "paul-gauguin.tar.gz"
#     # tar11 = "sample_photographs.tar.gz"
#     # tar12 = "samuel-peploe.tar.gz"
#     # tar13 = "vincent-van-gogh_road-with-cypresses-1890.tar.gz"
#     # tar14 = "wassily-kandinsky.tar.gz"
#     # places365_url = (
#     #     "data.csail.mit.edu/places/places365/train_large_places365standard.tar"
#     # )
#
#     return None


def dataset(root: str, transform: Optional[Transform] = None,) -> ImageFolderDataset:
    if transform is None:
        transform = image_transform()
    return ImageFolderDataset(root, transform=transform)


def batch_sampler(
    data_source: Sized,
    impl_params: bool = True,
    num_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> FiniteCycleBatchSampler:

    if num_batches is None:
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/main.py#L68
        num_batches = 300_000 if impl_params else 100_000

    if batch_size is None:
        batch_size = 1

    return FiniteCycleBatchSampler(
        data_source, num_batches=num_batches, batch_size=batch_size
    )


batch_sampler_ = batch_sampler


def image_loader(
    dataset: Dataset,
    impl_params: bool = True,
    batch_sampler: Optional[Sampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    if batch_sampler is None:
        batch_sampler = batch_sampler_(dataset, impl_params=impl_params)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
