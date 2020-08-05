from collections import OrderedDict
from typing import Any, Dict, Optional, Union, cast

import torch
from torch import nn

from pystiche import meta as meta_

from ..utils import join_channelwise

__all__ = [
    "Identity",
    "ResidualBlock",
    "SequentialWithOutChannels",
    "AddNoiseChannels",
    "HourGlassBlock",
]


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.residual = residual

        if shortcut is None:
            shortcut = Identity()
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.residual(x) + self.shortcut(x))


class SequentialWithOutChannels(nn.Sequential):
    def __init__(self, *args: Any, out_channel_name: Optional[Union[str, int]] = None):
        super().__init__(*args)
        if out_channel_name is None:
            out_channel_name = tuple(cast(Dict[str, nn.Module], self._modules).keys())[
                -1
            ]
        elif isinstance(out_channel_name, int):
            out_channel_name = str(out_channel_name)

        self.out_channels = cast(Dict[str, nn.Module], self._modules)[
            out_channel_name
        ].out_channels


class AddNoiseChannels(nn.Module):
    def __init__(
        self, in_channels: int, num_noise_channels: int = 3,
    ):
        super().__init__()
        self.num_noise_channels = num_noise_channels
        self.in_channels = in_channels
        self.out_channels = in_channels + num_noise_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        size = self._extract_size(input)
        meta = meta_.tensor_meta(input)
        noise = torch.rand(size, **meta)
        return join_channelwise(input, noise)

    def _extract_size(self, input: torch.Tensor) -> torch.Size:
        size = list(input.size())
        size[1] = self.num_noise_channels
        return torch.Size(size)


class HourGlassBlock(SequentialWithOutChannels):
    def __init__(
        self, downsample: nn.Module, intermediate: nn.Module, upsample: nn.Module
    ):
        modules = (
            ("down", downsample),
            ("intermediate", intermediate),
            ("up", upsample),
        )
        super().__init__(OrderedDict(modules), out_channel_name="intermediate")
