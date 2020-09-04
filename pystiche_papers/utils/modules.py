from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn

from ..utils import get_padding

__all__ = [
    "Identity",
    "ResidualBlock",
    "SequentialWithOutChannels",
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


class PaddedConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        padding: str = "valid",
        **kwargs: Any,
    ) -> None:
        self.use_correction_padding = False
        if self.is_any_even(kernel_size) and padding == "same":
            self.use_correction_padding = True
            self.correction_padding = self.get_padding_correction(kernel_size)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=get_padding(padding, kernel_size),
            **kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_correction_padding:
            input = nn.functional.pad(input, self.correction_padding)
        return cast(torch.Tensor, self._conv_forward(input, self.weight))

    def is_any_even(self, inputs: Union[Tuple[int, int], int]) -> bool:
        if isinstance(inputs, tuple):
            return any(input % 2 == 0 for input in inputs)
        return inputs % 2 == 0

    def get_padding_correction(
        self, kernel_size: Union[Tuple[int, int], int]
    ) -> List[int]:
        if isinstance(kernel_size, tuple):
            return list(
                sum(
                    tuple(
                        (0, 1) if self.is_any_even(size) else (0, 0)
                        for size in kernel_size
                    ),
                    (),
                )
            )
        return [0, 1, 0, 1]
