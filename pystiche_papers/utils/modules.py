from typing import Any, Dict, List, Optional, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd

import pystiche

__all__ = [
    "Identity",
    "ResidualBlock",
    "SequentialWithOutChannels",
    "SameSizeConv2d",
    "SameSizeConvTranspose2d",
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


class _SameSizeConvNdMixin(pystiche.ComplexObject, _ConvNd):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "padding" in kwargs:
            raise RuntimeError
        if "output_padding" in kwargs:
            raise RuntimeError
        super().__init__(*args, **kwargs)

        self._pad = self._compute_pad()
        self._mode = "constant" if self.padding_mode == "zeros" else self.padding_mode

    def _compute_pad(self) -> List[int]:
        kernel_size = torch.tensor(self.kernel_size)
        stride = torch.tensor(self.stride)
        dilation = torch.tensor(self.dilation)
        effective_kernel_size = dilation * (kernel_size - 1)

        if isinstance(self, _ConvTransposeNd):
            output_pad = torch.fmod(effective_kernel_size - 1, stride)
            pad_total = (effective_kernel_size - 1 - output_pad) // stride + 1

            self.padding = tuple(effective_kernel_size.tolist())
            self.output_padding = tuple(output_pad.tolist())
        else:
            pad_total = effective_kernel_size + 1 - stride

        pad_pre = pad_total // 2
        pad_post = pad_total - pad_pre
        return torch.stack((pad_pre, pad_post), dim=1).view(-1).flip(0).tolist()

    def forward(
        self, input: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        if output_size is not None:
            raise RuntimeError
        return cast(torch.Tensor, super().forward(F.pad(input, self._pad, self._mode)))

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["in_channels"] = self.in_channels
        dct["out_channels"] = self.out_channels
        dct["kernel_size"] = self.kernel_size
        dct["stride"] = self.stride
        if any(dilation != 1 for dilation in self.dilation):
            dct["dilation"] = self.dilation
        if self.groups != 1:
            dct["groups"] = self.groups
        if self.bias is None:
            dct["bias"] = True
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class SameSizeConv2d(_SameSizeConvNdMixin, nn.Conv2d):
    pass


class SameSizeConvTranspose2d(_SameSizeConvNdMixin, nn.ConvTranspose2d):
    pass
