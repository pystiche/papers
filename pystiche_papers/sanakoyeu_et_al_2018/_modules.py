import functools
from typing import List, Tuple, Union, cast

import more_itertools

import torch
from torch import nn

import pystiche
from pystiche.enc import SequentialEncoder
from pystiche.misc import verify_str_arg

from ..utils import ResidualBlock, same_size_padding

__all__ = [
    "get_padding",
    "get_activation",
    "conv",
    "ConvBlock",
    "UpsampleConvBlock",
    "residual_block",
    "encoder",
    "decoder",
    "Decoder",
    "Transformer",
]


def get_padding(
    padding: str, kernel_size: Union[Tuple[int, int], int]
) -> Union[Tuple[int, int], int]:
    padding = verify_str_arg(padding, valid_args=["same", "valid"])
    if padding == "same":
        return cast(Tuple[int, int], same_size_padding(kernel_size))
    else:  # padding == "valid"
        return 0


def get_activation(act: str = "relu", inplace: bool = True) -> nn.Module:
    act = verify_str_arg(act, valid_args=["relu", "lrelu"])
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    else:  # act == "lrelu"
        return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 2,
    padding: str = "valid",
) -> nn.Conv2d:
    padding = get_padding(padding, kernel_size)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding
    )


class ConvBlock(nn.Sequential):
    r"""ConvBlock from :cite:`SKL+2018`.

    This block comprises a convolution followed by a normalization and an optional
    activation.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Defaults to ``1``.
        padding: Padding of the input. It can be either ``"valid"`` for no padding or
            ``"same"`` for padding to preserve the size. Defaults to ``"valid"``.
        act: The activation is either ``"relu"`` for a :class:`~torch.nn.ReLU`,
            ``"lrelu"`` for a :class:`~torch.nn.LeakyReLU` with ``slope=0.2`` or
            ``None`` for no activation. Defaults to ``"relu"``.
        inplace: If ``True`` perform the activation in-place.

    The parameters ``kernel_size`` and ``stride`` can either be:
    * a single :class:`int` – in which case the same value is used for the height and
      width dimension
    * a tuple of two :class:`int` s – in which case, the first int is used for the
      vertical dimension, and the second int for the horizontal dimension
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        stride: Union[Tuple[int, int], int] = 1,
        padding: str = "valid",
        act: Union[str, None] = "relu",
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules: List[nn.Module] = []
        modules.append(
            conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        )

        modules.append(nn.InstanceNorm2d(out_channels))

        if act is not None:
            modules.append(get_activation(act=act, inplace=inplace))

        super().__init__(*modules)


class UpsampleConvBlock(nn.Module):
    r"""UpsampleConvBlock from :cite:`SKL+2018`.

    This block upsamples the input followed by a :class:`ConvBlock`.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        scale_factor: ``scale_factor`` of the interpolation. Defaults to ``2.0``.
        padding: Padding of the input. It can be either ``"valid"`` for no padding or
            ``"same"`` for padding to preserve the size. Defaults to ``"same"``.
        act: The activation is either ``"relu"`` for a :class:`~torch.nn.ReLU`,
            ``"lrelu"`` for a :class:`~torch.nn.LeakyReLU` with ``slope=0.2`` or
            ``None`` for no activation. Defaults to ``"relu"``.
        inplace: If ``True`` perform the activation in-place.

    The parameters ``kernel_size`` and ``stride`` can either be:
    * a single :class:`int` – in which case the same value is used for the height and
      width dimension
    * a tuple of two :class:`int` s – in which case, the first int is used for the
      vertical dimension, and the second int for the horizontal dimension
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        scale_factor: Union[Tuple[float, float], float] = 2.0,
        padding: str = "same",
        act: Union[str, None] = "relu",
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            act=act,
            inplace=inplace,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.conv(
                nn.functional.interpolate(
                    input, scale_factor=self.scale_factor, mode="nearest"
                )
            ),
        )


def residual_block(channels: int) -> ResidualBlock:
    r"""Residual block from :cite:`SKL+2018`.

    This block comprises two
    :class:`ConvBlock` without activation
    but respective prior reflection padding to maintain the input size as a ``residual``
    of a :class:`pystiche_papers.utils.modules.ResidualBlock`.

    Args:
        channels: Number of channels in the input.

    """
    in_channels = out_channels = channels
    kernel_size = 3
    padding = same_size_padding(kernel_size)

    residual = nn.Sequential(
        nn.ReflectionPad2d(padding),
        ConvBlock(
            in_channels, out_channels, kernel_size, stride=1, padding="valid", act=None,
        ),
        nn.ReflectionPad2d(padding),
        ConvBlock(
            in_channels, out_channels, kernel_size, stride=1, padding="valid", act=None,
        ),
    )

    return ResidualBlock(residual)


def encoder(in_channels: int = 3,) -> SequentialEncoder:
    r"""Encoder part of the :class:`Transformer` from :cite:`SKL+2018`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.

    """
    modules = [
        nn.ReflectionPad2d(15),
        ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1),
    ]
    channels = (32, 32, 64, 128, 256)

    functools.partial(
        modules.append,
        *[
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
            )
            for in_channels, out_channels in more_itertools.pairwise(channels)
        ],
    )

    return SequentialEncoder(modules)


def decoder(
    out_channels: int = 3, num_residual_blocks: int = 9,
) -> pystiche.SequentialModule:

    modules: List[nn.Module] = []
    for i in range(num_residual_blocks):
        modules.append(residual_block(256))

    channels = (256, 256, 128, 64, 32)
    functools.partial(
        modules.append,
        *[
            UpsampleConvBlock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=3
            )
            for in_channels, out_channels in more_itertools.pairwise(channels)
        ],
    )
    modules.append(nn.ReflectionPad2d(3))
    modules.append(
        conv(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding="valid",
        )
    )
    return pystiche.SequentialModule(*modules)


class Decoder(nn.Module):
    r"""Decoder part of the :class:`Transformer` from :cite:`SKL+2018`."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = decoder()
        self.output_module = self.ValueRangeDelimiter()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.output_module(self.decoder(input)))

    class ValueRangeDelimiter(nn.Module):
        r"""Maps the values to the interval (-1.0, 1.0)."""

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.tanh(input / 2)


class Transformer(nn.Module):
    r"""Transformer from :cite:`SKL+2018`."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = Decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(input)))
