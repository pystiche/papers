from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd

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


class _SameSizeConvNd(pystiche.Module):
    def __init__(self, conv_module: _ConvNd):
        if any(conv_module.padding):
            raise RuntimeError
        if any(conv_module.output_padding):
            raise RuntimeError

        super().__init__()
        self._conv = conv_module
        self._pad = self._compute_pad()
        self._mode = "constant" if self.padding_mode == "zeros" else self.padding_mode

    def _compute_pad(self) -> List[int]:
        kernel_size = torch.tensor(self.kernel_size)
        stride = torch.tensor(self.stride)
        dilation = torch.tensor(self.dilation)
        effective_kernel_size = dilation * (kernel_size - 1)

        if self._is_transpose_conv(self._conv):
            output_pad = torch.fmod(effective_kernel_size - 1, stride)
            pad_total = (effective_kernel_size - 1 - output_pad) // stride + 1

            self._conv.padding = tuple(effective_kernel_size.tolist())
            self._conv.output_padding = tuple(output_pad.tolist())
        else:
            pad_total = effective_kernel_size + 1 - stride

        pad_pre = pad_total // 2
        pad_post = pad_total - pad_pre
        return torch.stack((pad_pre, pad_post), dim=1).view(-1).flip(0).tolist()

    @staticmethod
    def _is_transpose_conv(conv_module: _ConvNd) -> bool:
        return isinstance(
            conv_module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv(F.pad(input, self._pad, mode=self._mode))

    def __getattr__(self, name: str) -> Any:
        if name == "_conv":
            return self._modules["_conv"]
        return getattr(self._conv, name)

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

    def _build_repr(
        self,
        name: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
        named_children: Optional[Sequence[Tuple[str, Any]]] = None,
    ) -> str:
        if named_children is None:
            named_children = tuple(
                (name, child)
                for name, child in self.named_children()
                if child is not self._conv
            )
        return super()._build_repr(
            name, properties=properties, named_children=named_children
        )


class SameSizeConv2d(_SameSizeConvNd):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(nn.Conv2d(*args, **kwargs))


class SameSizeConvTranspose2d(_SameSizeConvNd):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(nn.ConvTranspose2d(*args, **kwargs))
