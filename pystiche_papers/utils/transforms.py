from typing import cast

import torch
from torch import nn

# import pystiche.image.transforms.functional as F
from pystiche import image

# from pystiche.image import transforms


__all__ = [
    "OptionalGrayscaleToFakegrayscale",
]


class OptionalGrayscaleToFakegrayscale(nn.Module):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        is_grayscale = image.extract_num_channels(input_image) == 1
        if not is_grayscale:
            return input_image

        return input_image.repeat(1, 3, 1, 1)
