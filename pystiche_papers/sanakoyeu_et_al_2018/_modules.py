import functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche.misc import verify_str_arg
from pystiche_papers.utils import AutoPadAvgPool2d, AutoPadConv2d, channel_progression

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
    "TransformerBlock",
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
    bias: bool = False,
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
        bias=bias,
        **kwargs,
    )


def norm(
    channels: int,
    eps: float = 1e-5,
    affine: bool = True,
    track_running_stats: bool = False,
    momentum: float = 0.1,
) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(
        channels,
        eps=eps,
        affine=affine,
        track_running_stats=track_running_stats,
        momentum=momentum,
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
        padding: Optional Padding of the input. If ``None``, padding is done so that the
            output have the same spatial dimensions as the input. Defaults to ``0``.
        padding_mode: ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'zeros'``
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

        modules: List[nn.Module] = [
            conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
            ),
            norm(out_channels),
        ]

        if act is not None:
            modules.append(get_activation(act=act, inplace=inplace))

        super().__init__(*modules)


class UpsampleConvBlock(ConvBlock):
    r"""UpsampleConvBlock from :cite:`SKL+2018`.

    This block upsamples the input followed by a :class:`ConvBlock`.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        scale_factor: ``scale_factor`` of the interpolation. Defaults to ``2.0``.
        kwargs: Other optional arguments of :class:`ConvBlock`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        scale_factor: Union[Tuple[float, float], float] = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.scale_factor = scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(
            nn.functional.interpolate(
                input, scale_factor=self.scale_factor, mode="nearest"
            )
        )


def residual_block(channels: int, impl_params: bool = True) -> ResidualBlock:
    r"""Residual block from :cite:`SKL+2018`.

    This block comprises two
    :class:`ConvBlock` without activation
    but respective prior reflection padding to maintain the input size as a ``residual``
    of a :class:`pystiche_papers.utils.modules.ResidualBlock`.

    Args:
        channels: Number of channels in the input.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.


    If ``impl_params is True``, the first :class:`ConvBlock` uses a
    :class:`torch.nn.ReLU` activation.
    """
    conv_block = functools.partial(
        ConvBlock,
        channels,
        channels,
        kernel_size=3,
        stride=1,
        padding=None,
        padding_mode="reflect",
    )
    return ResidualBlock(
        nn.Sequential(
            # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/module.py#L81
            conv_block(act="relu" if impl_params else None),
            conv_block(act=None),
        )
    )


def encoder(impl_params: bool = True, in_channels: int = 3,) -> enc.SequentialEncoder:
    r"""Encoder part of the :class:`Transformer` from :cite:`SKL+2018`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see below.


    If ``impl_params is True``, an additional :class:`~torch.nn.InstanceNorm2d` layer
    is prefixed to the encoder.
    """
    modules: List[nn.Module] = []

    if impl_params:
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/module.py#L38-L40
        modules.append(norm(in_channels))
    modules.extend(
        (
            nn.ReflectionPad2d(15),
            ConvBlock(
                in_channels=in_channels, out_channels=32, kernel_size=3, stride=1
            ),
        )
    )
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
    impl_params: bool = True, out_channels: int = 3, num_residual_blocks: int = 9,
) -> pystiche.SequentialModule:
    r"""Decoder part of the :class:`Transformer` from :cite:`SKL+2018`."""
    residual_blocks = [
        residual_block(256, impl_params=impl_params) for _ in range(num_residual_blocks)
    ]
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
    def __init__(self, impl_params: bool = True) -> None:
        super().__init__()
        self.encoder = encoder(impl_params=impl_params)
        self.decoder = decoder(impl_params=impl_params)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(input)))


def transformer(style: Optional[str] = None, impl_params: bool = True) -> Transformer:
    r"""Transformer from :cite:`SKL+2018`.

    Args:
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`~pystiche_papers.sanakoyeu_et_al_2018.images`. If omitted, the
            transformer is initialized with random weights.
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    """
    return Transformer(impl_params=impl_params)


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


class TransformerBlock(enc.SequentialEncoder):
    r"""TransformerBlock from :cite:`SKL+2018`.

    This block takes an image as input and produce a transformed image of the same size.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
        kernel_size: Size of the convolving kernel. Defaults to ``10``.
        stride: Stride of the convolution. Defaults to ``1``.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.

    If ``impl_params is True``, an :class:`~torch.nn.AvgPool2d` is used instead of a
    :class:`~torch.nn.Conv2d` with :func:`~torch.nn.utils.weight_norm`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: Union[Tuple[int, int], int] = 10,
        stride: Union[Tuple[int, int], int] = 1,
        impl_params: bool = True,
    ):
        kwargs = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/module.py#L246
        module = (
            AutoPadAvgPool2d(**kwargs)
            if impl_params
            else nn.utils.weight_norm(AutoPadConv2d(in_channels, in_channels, **kwargs))
        )
        self.impl_params = impl_params
        super().__init__((module,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
