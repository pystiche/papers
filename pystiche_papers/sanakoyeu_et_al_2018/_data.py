import os
import re
from os import path
from typing import Any, Dict, List, Optional, Sized, Tuple, Union, cast
from urllib.parse import urljoin

import kornia
from kornia.augmentation.functional import apply_crop, compute_crop_transformation

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler
from torchvision.datasets.utils import download_and_extract_archive

from pystiche.data import (
    DownloadableImage,
    DownloadableImageCollection,
    ImageFolderDataset,
)
from pystiche.image import transforms
from pystiche.image.transforms import functional as F
from pystiche.image.utils import extract_edge_size, extract_image_size
from pystiche.misc import to_2d_arg, verify_str_arg

from ._augmentation import (
    AugmentationBase2d,
    _adapted_uniform,
    generate_vertices_from_size,
    post_crop_augmentation,
    pre_crop_augmentation,
)

__all__ = [
    "ClampSize",
    "OptionalUpsample",
    "RandomCrop",
    "image_transform",
    "images",
    "WikiArt",
    "style_dataset",
    "Places365Subset",
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


class OptionalUpsample(transforms.Transform):
    def __init__(
        self, min_edge_size: int, interpolation_mode: str = "bilinear"
    ) -> None:
        super().__init__()
        self.min_edge_size = min_edge_size
        self.interpolation_mode = interpolation_mode

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        edge_size = extract_edge_size(image, edge="short")
        if edge_size >= self.min_edge_size:
            return image

        return cast(
            torch.Tensor,
            F.resize(
                image,
                self.min_edge_size,
                edge="short",
                interpolation_mode=self.interpolation_mode,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["min_edge_size"] = self.min_edge_size
        if self.interpolation_mode != "bilinear":
            dct["interpolation_mode"] = self.interpolation_mode
        return dct


def _adapted_uniform_int(
    shape: Union[Tuple, torch.Size],
    low: Union[float, torch.Tensor],
    high: Union[float, torch.Tensor],
    same_on_batch: bool = False,
) -> torch.Tensor:
    return _adapted_uniform(shape, low, high + 1 - 1e-6, same_on_batch).int()


class RandomCrop(AugmentationBase2d):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L85-L87
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L129-L139
    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        p: float = 0.5,
        interpolation: str = "bilinear",
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = False,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform)
        self.size = cast(Tuple[int, int], to_2d_arg(size))
        self.interpolation = interpolation
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners
        self._flags = dict(
            interpolation=torch.tensor(kornia.Resample.get(interpolation).value),  # type: ignore[attr-defined]
            align_corners=torch.tensor(align_corners),
        )

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        batch_size = input_shape[0]
        image_size = cast(Tuple[int, int], input_shape[-2:])
        anchors = self.generate_anchors(
            batch_size, image_size, self.size, self.same_on_batch
        )
        dst = generate_vertices_from_size(batch_size, self.size)
        src = self.clamp_vertices_to_size(anchors + dst, image_size)
        return dict(src=src, dst=dst)

    @staticmethod
    def generate_anchors(
        batch_size: int,
        image_size: Tuple[int, int],
        crop_size: Tuple[int, int],
        same_on_batch: bool,
    ) -> torch.Tensor:
        def generate_single_dim_anchor(
            batch_size: int, image_length: int, crop_length: int, same_on_batch: bool
        ) -> torch.Tensor:
            diff = image_length - crop_length
            if diff <= 0:
                return torch.zeros((batch_size,), dtype=torch.int)
            else:
                return _adapted_uniform_int((batch_size,), 0, diff, same_on_batch)

        single_dim_anchors = [
            generate_single_dim_anchor(
                batch_size, image_length, crop_length, same_on_batch
            )
            for image_length, crop_length in zip(image_size, crop_size)
        ]
        return torch.stack(single_dim_anchors, dim=1,).unsqueeze(1).repeat(1, 4, 1)

    @staticmethod
    def clamp_vertices_to_size(
        vertices: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        horz, vert = vertices.split(1, dim=2)
        height, width = size
        return torch.cat(
            (torch.clamp(horz, 0, width - 1), torch.clamp(vert, 0, height - 1),), dim=2,
        )

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return cast(
            torch.Tensor, compute_crop_transformation(input, params, self._flags)
        )

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return apply_crop(input, params, self._flags)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
        if self.interpolation != "bilinear":
            dct["interpolation"] = self.interpolation
        if self.same_on_batch:
            dct["same_on_batch"] = True
        if self.align_corners:
            dct["align_corners"] = True
        return dct


def image_transform(impl_params: bool = True, edge_size: int = 768) -> nn.Sequential:
    transforms_: List[nn.Module] = [
        ClampSize() if impl_params else OptionalUpsample(edge_size),
    ]
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L286-L287
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L291-L292
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L271-L276
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L24
    if impl_params:
        transforms_.append(pre_crop_augmentation())
    transforms_.append(
        RandomCrop(edge_size, p=1.0)  # type: ignore[arg-type]
        if impl_params
        else transforms.ValidRandomCrop(edge_size)
    )
    if impl_params:
        transforms_.append(post_crop_augmentation())
    return nn.Sequential(*transforms_)


def url(
    id: int,
    file: str,
    base: str = "https://raw.githubusercontent.com/CompVis/adaptive-style-transfer/gh-pages/images/experiments/comparison_with_others/",
) -> str:
    return urljoin(base, f"{id:04d}/{file}")


def images() -> DownloadableImageCollection:
    content_images = {
        "garden": DownloadableImage(
            url(63, "Places365_val_00009975_content.jpg"),
            md5="f08cfeef2e020555e6bcf0f15d9d0e36",
        ),
        "bridge_river": DownloadableImage(
            url(65, "Places365_val_00010170_content.jpg"),
            md5="df290455d760e566b498878edb07e82c",
        ),
        "glacier_human": DownloadableImage(
            url(66, "Places365_val_00010262_content.jpg"),
            md5="420b014fc6b2e059a7c421ee5409495e",
        ),
        "mountain": DownloadableImage(
            url(74, "Places365_val_00010926_content.jpg"),
            md5="127ae1ad474a24a763d2f4f43b1c311e",
        ),
        "horses": DownloadableImage(
            url(76, "Places365_val_00010957_content.jpg"),
            md5="99658ffb5cea1275cf2411ba431a5b3c",
        ),
        "stone_facade": DownloadableImage(
            url(85, "Places365_val_00012263_content.jpg"),
            md5="8f4719f5f26ede9ea350102bd5a5a6e1",
        ),
        "waterway": DownloadableImage(
            url(86, "Places365_val_00012321_content.jpg"),
            md5="9b0e9aecf9cab7e956b3d3e999c0db0e",
        ),
        "garden_parc": DownloadableImage(
            url(87, "Places365_val_00012339_content.jpg"),
            md5="213680b20c63cfa981dcc0455276b79f",
        ),
    }

    return DownloadableImageCollection({**content_images})


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
    impl_params: bool = True,
    transform: Optional[nn.Module] = None,
    download: bool = False,
) -> WikiArt:
    if transform is None:
        transform = image_transform(impl_params=impl_params)
    return WikiArt(root, style, transform=transform, download=download)


# TODO: replace this with torchvision.datasets.Places365 as soon as
#  https://github.com/pytorch/vision/pull/2610 is part of a release
class Places365Subset(ImageFolderDataset):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/prepare_dataset.py#L80-L103
    CATEGORIES = (
        "/a/abbey",
        "/a/amphitheater",
        "/a/apartment-building/outdoor",
        "/a/aqueduct",
        "/a/arch",
        "/a/arena/rodeo",
        "/a/artists_loft",
        "/a/athletic_field/outdoor",
        "/b/badlands",
        "/b/balcony/exterior",
        "/b/bamboo_forest",
        "/b/barn",
        "/b/barndoor",
        "/b/baseball_field",
        "/b/basilica",
        "/b/bayou",
        "/b/beach",
        "/b/beach_house",
        "/b/beer_garden",
        "/b/boardwalk",
        "/b/boathouse",
        "/b/botanical_garden",
        "/b/building_facade",
        "/b/bullring",
        "/b/butte",
        "/c/cabin/outdoor",
        "/c/campsite",
        "/c/campus",
        "/c/canal/natural",
        "/c/canal/urban",
        "/c/canyon",
        "/c/castle",
        "/c/cemetery",
        "/c/chalet",
        "/c/church/outdoor",
        "/c/cliff",
        "/c/coast",
        "/c/corn_field",
        "/c/corral",
        "/c/cottage",
        "/c/courtyard",
        "/c/crevasse",
        "/d/dam",
        "/d/desert/vegetation",
        "/d/desert_road",
        "/d/doorway/outdoor",
        "/f/fairway",
        "/f/farm",
        "/f/field/cultivated",
        "/f/field/wild",
        "/f/field_road",
        "/f/fishpond",
        "/f/florist_shop/indoor",
        "/f/forest/broadleaf",
        "/f/forest_path",
        "/f/forest_road",
        "/f/formal_garden",
        "/g/gazebo/exterior",
        "/g/glacier",
        "/g/golf_course",
        "/g/gorge",
        "/g/greenhouse/indoor",
        "/g/greenhouse/outdoor",
        "/g/grotto",
        "/h/hayfield",
        "/h/herb_garden",
        "/h/hot_spring",
        "/h/house",
        "/h/hunting_lodge/outdoor",
        "/i/ice_floe",
        "/i/ice_shelf",
        "/i/iceberg",
        "/i/inn/outdoor",
        "/i/islet",
        "/j/japanese_garden",
        "/k/kasbah",
        "/k/kennel/outdoor",
        "/l/lagoon",
        "/l/lake/natural",
        "/l/lawn",
        "/l/library/outdoor",
        "/l/lighthouse",
        "/m/mansion",
        "/m/marsh",
        "/m/mausoleum",
        "/m/moat/water",
        "/m/mosque/outdoor",
        "/m/mountain",
        "/m/mountain_path",
        "/m/mountain_snowy",
        "/o/oast_house",
        "/o/ocean",
        "/o/orchard",
        "/p/park",
        "/p/pasture",
        "/p/pavilion",
        "/p/picnic_area",
        "/p/pier",
        "/p/pond",
        "/r/raft",
        "/r/railroad_track",
        "/r/rainforest",
        "/r/rice_paddy",
        "/r/river",
        "/r/rock_arch",
        "/r/roof_garden",
        "/r/rope_bridge",
        "/r/ruin",
        "/s/schoolhouse",
        "/s/sky",
        "/s/snowfield",
        "/s/swamp",
        "/s/swimming_hole",
        "/s/synagogue/outdoor",
        "/t/temple/asia",
        "/t/topiary_garden",
        "/t/tree_farm",
        "/t/tree_house",
        "/u/underwater/ocean_deep",
        "/u/utility_room",
        "/v/valley",
        "/v/vegetable_garden",
        "/v/viaduct",
        "/v/village",
        "/v/vineyard",
        "/v/volcano",
        "/w/waterfall",
        "/w/watering_hole",
        "/w/wave",
        "/w/wheat_field",
        "/z/zen_garden",
    )

    def _collect_image_files(self, depth: Optional[int]) -> Tuple[str, ...]:
        image_files = super()._collect_image_files(depth)

        categories = [
            category.replace("/", re.escape(os.sep)) for category in self.CATEGORIES
        ]
        pattern = re.compile(f"({'|'.join(categories)})")

        return tuple(
            image_file for image_file in image_files if pattern.search(image_file)
        )


def content_dataset(
    root: str, impl_params: bool = True, transform: Optional[nn.Module] = None,
) -> ImageFolderDataset:
    if transform is None:
        transform = image_transform(impl_params=impl_params)
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/prepare_dataset.py#L133
        if impl_params:
            transform = nn.Sequential(transforms.Rescale(2.0), transform)
    return Places365Subset(root, transform=transform)


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
