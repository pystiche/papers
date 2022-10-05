import torch
from torch import nn

from pystiche import image

__all__ = ["OptionalGrayscaleToFakegrayscale", "TopLeftCropToMultiple"]


class OptionalGrayscaleToFakegrayscale(nn.Module):
    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        is_grayscale = image.extract_num_channels(input_image) == 1
        if not is_grayscale:
            return input_image

        repeats = [1] * input_image.ndim
        repeats[-3] = 3
        return input_image.repeat(repeats)


class TopLeftCropToMultiple(nn.Module):
    def __init__(self, multiple: int = 16):
        super().__init__()
        self.multiple = multiple

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        old_height, old_width = image.extract_image_size(input_image)
        new_height = old_height - old_height % self.multiple
        new_width = old_width - old_width % self.multiple

        return input_image[..., :new_height, :new_width]
