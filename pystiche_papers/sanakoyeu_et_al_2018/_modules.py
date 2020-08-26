from typing import Any, Dict, List, Optional, Tuple, Union, cast
from collections import OrderedDict
from typing import Dict, List, Tuple, Union, cast

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche.misc import verify_str_arg
from pystiche_papers.utils import channel_progression

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
    "Transformer",
    "transformer",
    "get_transformation_block",
    "TransformerBlock",
    "DiscriminatorEncoder",
    "prediction_module",
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
        nn.ReflectionPad2d(3),
        conv(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding="valid",
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


def get_transformation_block(
    in_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int],
    impl_params: bool = True,
) -> nn.Module:
    if impl_params:
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return nn.Conv2d(in_channels, 3, kernel_size, stride=stride, padding=padding)


class TransformerBlock(nn.Module):
    r"""TransformerBlock from :cite:`SKL+2018`.

    This block takes an image as input and produce a transformed image of the same size.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
        kernel_size: Size of the convolving kernel. Defaults to ``10``.
        stride: Stride of the convolution. Defaults to ``1``.
        padding: Padding of the input. It can be either ``"valid"`` for no padding or
            ``"same"`` for padding to preserve the size. Defaults to ``"same"``.
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
        padding: str = "same",
        impl_params: bool = True,
    ):
        super().__init__()
        self.impl_params = impl_params

        padding = get_padding(padding, kernel_size)

        self.forwardBlock = get_transformation_block(
            in_channels, kernel_size, stride, padding, impl_params=impl_params,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.impl_params:
            return cast(torch.Tensor, self.forwardBlock(input))
        else:
            return cast(torch.Tensor, nn.utils.weight_norm(self.forwardBlock(input)))


def discriminator_encoder_modules(
    in_channels: int = 3, inplace: bool = True,
) -> Dict[str, nn.Sequential]:
    # FIXME:  if/when the Python interpreter will learn to accept the correct signature
    # https://stackoverflow.com/questions/41207128/how-do-i-specify-ordereddict-k-v-types-for-mypy-type-annotation

    modules = OrderedDict(
        {
            "scale_0": ConvBlock(
                in_channels,
                128,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_1": ConvBlock(
                128,
                128,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_2": ConvBlock(
                128,
                256,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_3": ConvBlock(
                256,
                512,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_4": ConvBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_5": ConvBlock(
                512,
                1024,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
            "scale_6": ConvBlock(
                1024,
                1024,
                kernel_size=5,
                stride=2,
                padding="same",
                act="lrelu",
                inplace=inplace,
            ),
        }
    )
    return cast(Dict[str, nn.Sequential], modules)


class DiscriminatorEncoder(enc.MultiLayerEncoder):
    r"""Encoder part of the Discriminator from :cite:`SKL+2018`.

    Args:
        in_channels: Number of channels in the input. Defaults to ``3``.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__(
            self._collect_modules(
                discriminator_encoder_modules(in_channels=in_channels)
            )
        )

    def _collect_modules(
        self, wrapped_modules: Dict[str, nn.Sequential]
    ) -> List[Tuple[str, nn.Module]]:
        modules = []
        block = 0
        for sequential in wrapped_modules.values():
            for module in sequential._modules.values():
                if isinstance(module, nn.Conv2d):
                    name = f"conv{block}"
                elif isinstance(module, nn.InstanceNorm2d):
                    name = f"inst_n{block}"
                else:  # isinstance(module, nn.LeakyReLU):
                    name = f"lrelu{block}"
                    # each LeakyReLU layer marks the end of the current block
                    block += 1

                modules.append((name, module))
        return modules


def prediction_module(
    in_channels: int, kernel_size: Union[Tuple[int, int], int], padding: str = "same"
) -> nn.Module:
    r"""Prediction module from :cite:`SKL+2018`.

    This block comprises a convolutional, which is used as an auxiliary classifier to
    capture image details on different scales of the
    :class:`~pystiche_paper.sanakoyeu_et_al_2018._modules.DiscriminatorEncoder`.

    Args:
        in_channels: Number of channels in the input.
        kernel_size: Size of the convolving kernel.
        padding: Padding of the input. It can be either be``"valid"`` for no padding or
            ``"same"`` to keep the size. Defaults to ``"same"``.

    """
    return conv(
        in_channels=in_channels,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
