from typing import Tuple, cast

import torch

import pystiche.image.transforms.functional as F
from pystiche import image
from pystiche.image import transforms

__all__ = [
    "TopLeftCropToMultiple",
    "OptionalGrayscaleToFakegrayscale",
    "MirrorHorizontally",
]


class TopLeftCropToMultiple(transforms.Transform):
    def __init__(self, multiple: int):
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


class OptionalGrayscaleToFakegrayscale(transforms.Transform):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        is_grayscale = image.extract_num_channels(input_image) == 1
        if is_grayscale:
            return cast(torch.Tensor, F.grayscale_to_fakegrayscale(input_image))
        else:
            return input_image


class MirrorHorizontally(transforms.Transform):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.flip(2)
