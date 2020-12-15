from os import path
from typing import Any, Dict, List, Optional, cast

import torch
from torch import nn

import pystiche
from pystiche import enc
from pystiche_papers.sanakoyeu_et_al_2018._modules import (
    conv_block,
    UpsampleConvBlock,
    conv,
    norm,
    residual_block,
)
from pystiche_papers.utils import (
    channel_progression,
    load_state_dict_from_url,
    select_url_from_csv,
    str_to_bool,
)

__all__ = [
    "encoder",
    "decoder",
    "Transformer",
    "transformer",
]


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
            conv_block(
                in_channels=in_channels, out_channels=32, kernel_size=3, stride=1
            ),
        )
    )
    modules.extend(
        channel_progression(
            lambda in_channels, out_channels: conv_block(
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
            in_channels, out_channels, kernel_size=3, padding=None,
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
    def __init__(self, impl_params: bool = True, init_weights: bool = True) -> None:
        super().__init__()
        self.encoder = encoder(impl_params=impl_params)
        self.decoder = decoder(impl_params=impl_params)
        if init_weights:
            self.init_weights(impl_params=impl_params)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(input)))

    def init_weights(self, impl_params: bool = True) -> None:
        if not impl_params:
            return

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/ops.py#L54
                # https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/initializers/truncated_normal
                std = 0.02
                nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
                )
            if isinstance(module, nn.InstanceNorm2d):
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/ops.py#L42-L43
                # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/random_normal_initializer
                nn.init.normal_(module.weight, mean=1.0, std=0.02)
                nn.init.zeros_(module.bias)


# The Tensorflow weights were created by Artsiom Sanakoyeu, Dmytro Kotovenko,
# Sabine Lang, and Björn Ommer. See https://download.pystiche.org/models/LICENSE.md for
# details.
def select_url(framework: str, style: str, impl_params: bool) -> str:
    return select_url_from_csv(
        path.join(path.dirname(__file__), "model_urls.csv"),
        (framework, style, impl_params),
        converters=dict(impl_params=str_to_bool),
    )


def transformer(
    style: Optional[str] = None, impl_params: bool = True, framework: str = "pystiche",
) -> Transformer:
    r"""Pretrained transformer from :cite:`SKL+2018` .

    Args:
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`~pystiche_papers.johnson_alahi_li_2016.images`. If omitted, the
            transformer is initialized with random weights according to the procedure
            used by the original authors.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        framework: Framework that was used to train the the transformer. Can be one of
            ``"pystiche"`` (default) and ``"tensorflow"``.
    """
    init_weights = style is None
    transformer_ = Transformer(impl_params=impl_params, init_weights=init_weights)
    if init_weights:
        return transformer_

    url = select_url(
        framework=framework, style=cast(str, style), impl_params=impl_params,
    )
    state_dict = load_state_dict_from_url(url)
    transformer_.load_state_dict(state_dict)
    return transformer_
