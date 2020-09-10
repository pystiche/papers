from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche.misc import verify_str_arg
from pystiche_papers.utils import AutoPadConv2d, channel_progression

from ..utils import ResidualBlock

__all__ = [
    "get_activation",
    "conv",
    "ConvBlock",
    "UpsampleConvBlock",
    "residual_block",
    "encoder",
    "decoder",
    "Transformer",
    "transformer",
    "Discriminator",
    "DiscriminatorMultiLayerEncoder",
]


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
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[int] = 0,
    padding_mode: str = "zeros",
) -> nn.Conv2d:
    cls: Type[nn.Conv2d]
    kwargs: Dict[str, Any]
    cls, kwargs = (
        (AutoPadConv2d, {}) if padding is None else (nn.Conv2d, dict(padding=padding))
    )
    return cls(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding_mode=padding_mode,
        **kwargs,
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
        padding: Optional[int] = 0,
        padding_mode: str = "zeros",
        act: Optional[str] = "relu",
        inplace: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels

        modules: List[nn.Module] = []
        modules.append(
            conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            )
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
        stride: Union[Tuple[int, int], int] = 1,
        scale_factor: Union[Tuple[float, float], float] = 2.0,
        padding: Optional[int] = None,
        act: Union[str, None] = "relu",
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
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
    return ResidualBlock(
        nn.Sequential(
            *[
                ConvBlock(
                    channels,
                    channels,
                    kernel_size=3,
                    stride=1,
                    padding=None,
                    padding_mode="reflect",
                    act=None,
                )
                for _ in range(2)
            ]
        )
    )


def encoder(in_channels: int = 3,) -> enc.SequentialEncoder:
    r"""Encoder part of the :class:`Transformer` from :cite:`SKL+2018`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.

    """
    modules = [
        nn.ReflectionPad2d(15),
        ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1),
    ]
    modules.extend(
        channel_progression(
            lambda in_channels, out_channels: ConvBlock(
                in_channels, out_channels, kernel_size=3, stride=2
            ),
            channels=(32, 32, 64, 128, 256),
        )
    )
    return enc.SequentialEncoder(modules)


class ValueRangeDelimiter(pystiche.Module):
    r"""Maps the values to the interval (-1.0, 1.0)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input / 2)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["output_range"] = (-1.0, 1.0)
        return dct


def decoder(
    out_channels: int = 3, num_residual_blocks: int = 9,
) -> pystiche.SequentialModule:
    r"""Decoder part of the :class:`Transformer` from :cite:`SKL+2018`."""
    residual_blocks = [residual_block(256) for _ in range(num_residual_blocks)]
    upsample_conv_blocks = channel_progression(
        lambda in_channels, out_channels: UpsampleConvBlock(
            in_channels, out_channels, kernel_size=3
        ),
        channels=(256, 256, 128, 64, 32),
    )
    return pystiche.SequentialModule(
        *residual_blocks,
        *upsample_conv_blocks,
        conv(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding=None,
            padding_mode="reflect",
        ),
        ValueRangeDelimiter(),
    )


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(input)))


def transformer(style: Optional[str] = None,) -> Transformer:
    r"""Transformer from :cite:`SKL+2018`.

    Args:
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.images`. If omitted, the
            transformer is initialized with random weights.

    """
    return Transformer()


class Discriminator(pystiche.Module):
    r"""Discriminator from :cite:`SKL+2018`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            indexed_children=channel_progression(
                lambda in_channels, out_channels: ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=None,
                    act="lrelu",
                ),
                channels=(in_channels, 128, 128, 256, 512, 512, 1024, 1024),
            )
        )


class DiscriminatorMultiLayerEncoder(enc.MultiLayerEncoder):
    r"""Discriminator from :cite:`SKL+2018` as :class:`pystiche.enc.MultiLayerEncoder`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(tuple(Discriminator(in_channels=in_channels).named_children()))
