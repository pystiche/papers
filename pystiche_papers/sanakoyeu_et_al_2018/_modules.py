import functools
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from torch import nn

from pystiche import enc
from pystiche.misc import verify_str_arg
from pystiche_papers.utils import AutoPadAvgPool2d, AutoPadConv2d

from ..utils import ResidualBlock

__all__ = [
    "get_activation",
    "conv",
    "ConvBlock",
    "UpsampleConvBlock",
    "residual_block",
    "TransformerBlock",
]


# TODO: rename to activation
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


# TODO: create a function called conv_block instead of creating directly
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


# TODO: create a function called upsample_conv_block instead of creating directly
# TODO: (distant future) merge UpsampleConvBlock with ConvBlock
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
        interpolated_input = nn.functional.interpolate(
            input, scale_factor=self.scale_factor, mode="nearest"
        )
        return cast(torch.Tensor, super().forward(interpolated_input))


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


# TODO: since this is only used in the transformed image loss, shouldn't it be defined
#  there?
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
