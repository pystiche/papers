import csv
from math import sqrt
from os import path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn

import pystiche
from pystiche_papers.utils import load_state_dict_from_url

from ..utils import ResidualBlock, same_size_output_padding, same_size_padding

__all__ = [
    "conv",
    "norm",
    "conv_block",
    "residual_block",
    "encoder",
    "decoder",
    "Transformer",
    "transformer",
]


def _load_model_urls() -> Dict[Tuple[str, str, bool, bool], str]:
    def str_to_bool(string: str) -> bool:
        return string.lower() == "true"

    with open(path.join(path.dirname(__file__), "model_urls.csv"), "r") as fh:
        return {
            (
                row["framework"],
                row["style"],
                str_to_bool(row["impl_params"]),
                str_to_bool(row["instance_norm"]),
            ): row["url"]
            for row in csv.DictReader(fh)
        }


# The LuaTorch weights were created by Justin Johnson, Alexandre Alahi, and Fei-Fei Li.
# See https://download.pystiche.org/models/LICENSE for details.
MODEL_URLS = _load_model_urls()


def select_url(
    framework: str, style: str, impl_params: bool, instance_norm: bool
) -> str:
    try:
        return MODEL_URLS[(framework, style, impl_params, instance_norm)]
    except KeyError:
        msg = (
            f"No pre-trained weights available for the parameter configuration\n\n"
            f"framework: {framework}\nstyle: {style}\nimpl_params: {impl_params}\n"
            f"instance_norm: {instance_norm}"
        )
        raise RuntimeError(msg)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
) -> Union[nn.Conv2d, nn.ConvTranspose2d]:
    if padding is None:
        padding = cast(Union[Tuple[int, int], int], same_size_padding(kernel_size))
    if not upsample:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        )
    else:
        output_padding = cast(
            Union[Tuple[int, int], int], same_size_output_padding(stride)
        )
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )


def norm(
    out_channels: int, instance_norm: bool
) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    norm_kwargs: Dict[str, Any] = {
        "eps": 1e-5,
        "momentum": 1e-1,
        "affine": True,
        "track_running_stats": True,
    }
    if instance_norm:
        return nn.InstanceNorm2d(out_channels, **norm_kwargs)
    else:
        return nn.BatchNorm2d(out_channels, **norm_kwargs)


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int] = 1,
    padding: Optional[Union[Tuple[int, int], int]] = None,
    upsample: bool = False,
    relu: bool = True,
    inplace: bool = True,
    instance_norm: bool = True,
) -> nn.Sequential:
    modules: List[nn.Module] = [
        conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
        ),
        norm(out_channels, instance_norm),
    ]
    if relu:
        modules.append(nn.ReLU(inplace=inplace))
    return nn.Sequential(*modules)


def residual_block(
    channels: int, inplace: bool = True, instance_norm: bool = True,
) -> ResidualBlock:
    in_channels = out_channels = channels
    kernel_size = 3
    residual = nn.Sequential(
        conv_block(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            inplace=inplace,
            instance_norm=instance_norm,
        ),
        conv_block(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            instance_norm=instance_norm,
            relu=False,
        ),
    )

    class CenterCrop(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, 2:-2, 2:-2]

    shortcut = CenterCrop()

    return ResidualBlock(residual, shortcut)


def maybe_fix_num_channels(num_channels: int, instance_norm: bool) -> int:
    return num_channels if not instance_norm else num_channels // 2


def encoder(instance_norm: bool = True,) -> pystiche.SequentialModule:
    modules = (
        nn.ReflectionPad2d(40),
        conv_block(
            in_channels=3,
            out_channels=maybe_fix_num_channels(32, instance_norm),
            kernel_size=9,
            instance_norm=instance_norm,
        ),
        conv_block(
            in_channels=maybe_fix_num_channels(32, instance_norm),
            out_channels=maybe_fix_num_channels(64, instance_norm),
            kernel_size=3,
            stride=2,
            instance_norm=instance_norm,
        ),
        conv_block(
            in_channels=maybe_fix_num_channels(64, instance_norm),
            out_channels=maybe_fix_num_channels(128, instance_norm),
            kernel_size=3,
            stride=2,
            instance_norm=instance_norm,
        ),
        residual_block(channels=maybe_fix_num_channels(128, instance_norm)),
        residual_block(channels=maybe_fix_num_channels(128, instance_norm)),
        residual_block(channels=maybe_fix_num_channels(128, instance_norm)),
        residual_block(channels=maybe_fix_num_channels(128, instance_norm)),
        residual_block(channels=maybe_fix_num_channels(128, instance_norm)),
    )
    return pystiche.SequentialModule(*modules)


def decoder(
    impl_params: bool = True, instance_norm: bool = True,
) -> pystiche.SequentialModule:
    def get_value_range_delimiter() -> nn.Module:
        if impl_params:

            def value_range_delimiter(x: torch.Tensor) -> torch.Tensor:
                return 150.0 * torch.tanh(x)

        else:

            def value_range_delimiter(x: torch.Tensor) -> torch.Tensor:
                # (tanh(x) + 1) / 2 == sgm(2*x)
                return torch.sigmoid(2.0 * x)

        class ValueRangeDelimiter(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return value_range_delimiter(x)

        return ValueRangeDelimiter()

    modules = (
        conv_block(
            in_channels=maybe_fix_num_channels(128, instance_norm),
            out_channels=maybe_fix_num_channels(64, instance_norm),
            kernel_size=3,
            stride=2,
            upsample=True,
            instance_norm=instance_norm,
        ),
        conv_block(
            in_channels=maybe_fix_num_channels(64, instance_norm),
            out_channels=maybe_fix_num_channels(32, instance_norm),
            kernel_size=3,
            stride=2,
            upsample=True,
            instance_norm=instance_norm,
        ),
        nn.Conv2d(
            in_channels=maybe_fix_num_channels(32, instance_norm),
            out_channels=3,
            kernel_size=9,
            padding=same_size_padding(kernel_size=9),
        ),
        get_value_range_delimiter(),
    )

    return pystiche.SequentialModule(*modules)


class Transformer(nn.Module):
    def __init__(
        self,
        impl_params: bool = True,
        instance_norm: bool = True,
        init_weights: bool = True,
    ):
        super().__init__()
        self.encoder = encoder(instance_norm=instance_norm)
        self.decoder = decoder(impl_params=impl_params, instance_norm=instance_norm)
        if init_weights:
            self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / sqrt(fan_in)
                nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -bound, bound)
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(x)))


def transformer(
    style: Optional[str] = None,
    framework: str = "pystiche",
    impl_params: bool = True,
    instance_norm: bool = True,
) -> Transformer:
    r"""Pretrained transformer from :cite:`JAL2016` .

    Args:
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`pystiche_papers.johnson_alahi_li_2016.images`. If omitted, the
            transformer is initialized with random weights according to the procedure
            used by the original authors.
        framework: Framework that was used to train the the transformer. Can be one of
            ``"pystiche"`` (default) and ``"luatorch"``.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper.

    For ``framework == "pystiche"`` all combinations of parameters are available.

    The weights for ``framework == "luatorch"`` were ported from the reference
    implementation (``impl_params is True``) of the original authors. See
    https://download.pystiche.org/models/LICENSE for licensing details. The following
    combinations of parameters are available:

    +-------------------------------+----------+-----------+
    | ``style``                     | ``instance_norm``    |
    +-------------------------------+----------+-----------+
    |                               | ``True`` | ``False`` |
    +===============================+==========+===========+
    | ``"candy"``                   | x        |           |
    +-------------------------------+----------+-----------+
    | ``"composition_vii"``         |          | x         |
    +-------------------------------+----------+-----------+
    | ``"feathers"``                | x        |           |
    +-------------------------------+----------+-----------+
    | ``"la_muse"``                 | x        | x         |
    +-------------------------------+----------+-----------+
    | ``"mosaic"``                  | x        |           |
    +-------------------------------+----------+-----------+
    | ``"starry_night"``            |          | x         |
    +-------------------------------+----------+-----------+
    | ``"the_scream"``              | x        |           |
    +-------------------------------+----------+-----------+
    | ``"the_wave"``                |          | x         |
    +-------------------------------+----------+-----------+
    | ``"udnie"``                   | x        |           |
    +-------------------------------+----------+-----------+
    """
    init_weights = style is None
    transformer_ = Transformer(
        impl_params=impl_params, instance_norm=instance_norm, init_weights=init_weights
    )
    if init_weights:
        return transformer_

    url = select_url(
        framework=framework,
        style=cast(str, style),
        impl_params=impl_params,
        instance_norm=instance_norm,
    )
    state_dict = load_state_dict_from_url(url)
    transformer_.load_state_dict(state_dict)
    return transformer_
