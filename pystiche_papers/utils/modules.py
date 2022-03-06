import functools
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import more_itertools

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _AvgPoolNd

import pystiche
from pystiche.misc import to_2d_arg

__all__ = [
    "Identity",
    "ResidualBlock",
    "SequentialWithOutChannels",
    "AutoPadConv2d",
    "AutoPadConvTranspose2d",
    "AutoPadAvgPool2d",
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
            out_channel_name = tuple(self._modules.keys())[-1]
        elif isinstance(out_channel_name, int):
            out_channel_name = str(out_channel_name)

        self.out_channels = self._modules[out_channel_name].out_channels


class _AutoPadNdMixin(pystiche.ComplexObject):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "padding" in kwargs:
            raise RuntimeError
        super().__init__(*args, **kwargs)

        self._pad = self._pad_size_to_pad(self._compute_pad_size())

    @abstractmethod
    def _compute_pad_size(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def _mode(self) -> str:
        pass

    @staticmethod
    def _pad_size_to_pad(size: torch.Tensor) -> List[int]:
        pad_post = torch.div(size, 2, rounding_mode="floor")
        pad_pre = size - pad_post
        return torch.stack((pad_pre, pad_post), dim=1).view(-1).flip(0).tolist()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            super().forward(F.pad(input, self._pad, self._mode)),  # type: ignore[misc]
        )


class _AutoPadConvNdMixin(_AutoPadNdMixin, _ConvNd):
    def _compute_pad_size(self) -> torch.Tensor:
        kernel_size = torch.tensor(self.kernel_size)
        stride = torch.tensor(self.stride)
        dilation = torch.tensor(self.dilation)
        effective_kernel_size = dilation * (kernel_size - 1)
        return effective_kernel_size + 1 - stride

    @property
    def _mode(self) -> str:
        return "constant" if self.padding_mode == "zeros" else self.padding_mode

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
            dct["bias"] = False
        if self.padding_mode != "zeros":
            dct["padding_mode"] = self.padding_mode
        return dct


class AutoPadConv2d(_AutoPadConvNdMixin, nn.Conv2d):
    pass


class _AutoPadConvTransposeNdMixin(_AutoPadConvNdMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "output_padding" in kwargs:
            raise RuntimeError
        super().__init__(*args, **kwargs)

    def forward(
        self, input: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        if output_size is not None:
            raise RuntimeError
        return super().forward(input)

    def _compute_pad_size(self) -> torch.Tensor:
        kernel_size = torch.tensor(self.kernel_size)
        stride = torch.tensor(self.stride)
        dilation = torch.tensor(self.dilation)
        effective_kernel_size = dilation * (kernel_size - 1)

        self.padding = tuple(effective_kernel_size.tolist())

        output_pad = torch.fmod(effective_kernel_size - 1, stride)
        self.output_padding = tuple(output_pad.tolist())

        pad_size = (
            torch.div(
                effective_kernel_size - 1 - output_pad, stride, rounding_mode="floor",
            )
            + 1
        )

        return cast(torch.Tensor, pad_size)


class AutoPadConvTranspose2d(  # type: ignore[misc]
    _AutoPadConvTransposeNdMixin, nn.ConvTranspose2d,
):
    pass


class _AutoPadAvgPoolNdMixin(_AutoPadNdMixin, _AvgPoolNd):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.count_include_pad:
            if len(cast(Tuple[int, ...], self.kernel_size)) > 2:
                raise RuntimeError(
                    f"count_include_pad=False is not yet supported for {type(self)}"
                )
            if any(stride != 1 for stride in cast(Tuple[int, ...], self.stride)):
                raise RuntimeError(
                    "count_include_pad=False is not yet supported for strides > 1"
                )

    def _compute_pad_size(self) -> torch.Tensor:
        kernel_size = torch.tensor(self.kernel_size)
        stride = torch.tensor(self.stride)
        return torch.max(kernel_size - stride, torch.zeros_like(kernel_size))

    @property
    def _mode(self) -> str:
        return "constant"

    @staticmethod
    @functools.lru_cache()
    def _compute_count_correction(
        size: Tuple[int, ...],
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        pad: Tuple[int, ...],
    ) -> torch.Tensor:
        corrections: List[torch.Tensor] = []
        for size_, kernel_size_, stride_, pad_ in zip(
            size, kernel_size, stride, more_itertools.chunked(pad, 2)
        ):
            pre = kernel_size_ / torch.arange(
                start=(kernel_size_ - pad_[0]),
                end=(kernel_size_ - 1) + 1,
                step=1,
                dtype=torch.float,
            )
            post = kernel_size_ / torch.arange(
                start=kernel_size_ - 1,
                end=(kernel_size_ - pad_[1]) - 1,
                step=-1,
                dtype=torch.float,
            )
            intermediate = torch.ones(size_ - sum(pad_))
            corrections.append(torch.cat((pre, intermediate, post)))

        if len(corrections) == 1:
            return corrections[0]
        else:  # len(corrections) == 2
            return torch.mm(corrections[0].unsqueeze(1), corrections[1].unsqueeze(0))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.count_include_pad:
            return output

        count_correction = self._compute_count_correction(
            input.size()[2:], self.kernel_size, self.stride, tuple(self._pad)
        )
        return output * count_correction.to(output)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["kernel_size"] = self.kernel_size
        dct["stride"] = self.stride
        return dct


class AutoPadAvgPool2d(_AutoPadAvgPoolNdMixin, nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        stride: Optional[Union[Tuple[int, int], int]] = None,
        **kwargs: Any,
    ) -> None:
        kernel_size = to_2d_arg(kernel_size)
        stride = kernel_size if stride is None else to_2d_arg(stride)
        super().__init__(kernel_size, stride=stride, **kwargs)
