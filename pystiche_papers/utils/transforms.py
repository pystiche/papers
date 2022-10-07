import torch
from torch import nn

from pystiche import image

__all__ = [
    "OptionalGrayscaleToFakegrayscale"
]


class OptionalGrayscaleToFakegrayscale(nn.Module):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        is_grayscale = image.extract_num_channels(input_image) == 1
        if not is_grayscale:
            return input_image

        repeats = [1] * input_image.ndim
        repeats[-3] = 3
        return input_image.repeat(repeats)
