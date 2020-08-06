from typing import cast

import torch

import pystiche.image.transforms.functional as F
from pystiche import image
from pystiche.image import transforms

__all__ = [
    "OptionalGrayscaleToFakegrayscale",
]


class OptionalGrayscaleToFakegrayscale(transforms.Transform):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        is_grayscale = image.extract_num_channels(input_image) == 1
        if is_grayscale:
            return cast(torch.Tensor, F.grayscale_to_fakegrayscale(input_image))
        else:
            return input_image
