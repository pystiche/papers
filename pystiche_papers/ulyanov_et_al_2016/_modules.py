from collections import OrderedDict
from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn

from pystiche import meta as meta_
from pystiche import misc

from ..utils import is_valid_padding, same_size_padding


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


def join_channelwise(*inputs: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    return torch.cat(inputs, dim=channel_dim)


class AddNoiseChannels(nn.Module):
    def __init__(
        self, in_channels: int, num_noise_channels: int = 3,
    ):
        super().__init__()
        self.num_noise_channels = num_noise_channels
        self.in_channels = in_channels
        self.out_channels = in_channels + num_noise_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        size = self._extract_size(input)
        meta = meta_.tensor_meta(input)
        noise = torch.rand(size, **meta)
        return join_channelwise(input, noise)

    def _extract_size(self, input: torch.Tensor) -> torch.Size:
        size = list(input.size())
        size[1] = self.num_noise_channels
        return torch.Size(size)


def noise(in_channels: int = 3, num_noise_channels: int = 3,) -> AddNoiseChannels:
    return AddNoiseChannels(in_channels, num_noise_channels=num_noise_channels)


def downsample(kernel_size: int = 2, stride: int = 2, padding: int = 0) -> nn.AvgPool2d:
    return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)


def upsample() -> nn.Upsample:
    return nn.Upsample(scale_factor=2.0, mode="nearest")


class HourGlassBlock(SequentialWithOutChannels):
    def __init__(self, intermediate: nn.Module):
        modules = (
            ("down", downsample()),
            ("intermediate", intermediate),
            ("up", upsample()),
        )
        super().__init__(OrderedDict(modules), out_channel_name="intermediate")


def norm(
    in_channels: int, instance_norm: bool
) -> Union[nn.BatchNorm2d, nn.InstanceNorm2d]:
    norm_kwargs: Dict[str, Any] = {
        "eps": 1e-5,
        "momentum": 1e-1,
        "affine": True,
        "track_running_stats": True,
    }
    return (
        nn.InstanceNorm2d(in_channels, **norm_kwargs)
        if instance_norm
        else nn.BatchNorm2d(in_channels, **norm_kwargs)
    )


def activation(
    impl_params: bool, instance_norm: bool, inplace: bool = True
) -> nn.Module:
    return (
        nn.ReLU(inplace=inplace)
        if impl_params and instance_norm
        else nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    )


class ConvBlock(SequentialWithOutChannels):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int],
        impl_params: bool = True,
        stride: Union[Tuple[int, int], int] = 1,
        instance_norm: bool = True,
        inplace: bool = True,
    ) -> None:
        padding = cast(
            Union[Tuple[int, int, int, int], int], same_size_padding(kernel_size)
        )

        modules: List[Tuple[str, nn.Module]] = []

        if is_valid_padding(padding):
            modules.append(("pad", nn.ReflectionPad2d(padding)))

        modules.append(
            (
                "conv",
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=0
                ),
            )
        )
        modules.append(("norm", norm(out_channels, instance_norm)))
        modules.append(("act", activation(impl_params, instance_norm, inplace=inplace)))

        super().__init__(OrderedDict(modules), out_channel_name="conv")


class ConvSequence(SequentialWithOutChannels):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        inplace: bool = True,
    ):
        def conv_block(
            in_channels: int, out_channels: int, kernel_size: int
        ) -> ConvBlock:
            return ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                impl_params=impl_params,
                instance_norm=instance_norm,
                inplace=inplace,
            )

        modules = (
            ("conv_block1", conv_block(in_channels, out_channels, kernel_size=3)),
            ("conv_block2", conv_block(out_channels, out_channels, kernel_size=3)),
            ("conv_block3", conv_block(out_channels, out_channels, kernel_size=1)),
        )

        super().__init__(OrderedDict(modules))


class JoinBlock(nn.Module):
    def __init__(
        self,
        branch_in_channels: Sequence[int],
        names: Optional[Sequence[str]] = None,
        instance_norm: bool = True,
        channel_dim: int = 1,
    ) -> None:
        super().__init__()

        num_branches = len(branch_in_channels)
        if names is None:
            names = [str(idx) for idx in range(num_branches)]
        else:
            if len(names) != num_branches:
                raise RuntimeError

        norm_modules = [
            norm(in_channels, instance_norm) for in_channels in branch_in_channels
        ]

        for name, module in zip(names, norm_modules):
            self.add_module(name, module)

        self.norm_modules = norm_modules
        self.channel_dim = channel_dim

    @property
    def out_channels(self) -> int:
        return sum([norm.num_features for norm in self.norm_modules])

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return join_channelwise(
            *[norm(input) for norm, input in misc.zip_equal(self.norm_modules, inputs)],
            channel_dim=self.channel_dim,
        )


class BranchBlock(nn.Module):
    def __init__(
        self,
        deep_branch: nn.Module,
        shallow_branch: nn.Module,
        instance_norm: bool = True,
    ):
        super().__init__()
        self.deep = deep_branch
        self.shallow = shallow_branch
        self.join = JoinBlock(
            (
                cast(int, deep_branch.out_channels),
                cast(int, shallow_branch.out_channels),
            ),
            ("deep", "shallow"),
            instance_norm=instance_norm,
        )

    @property
    def out_channels(self) -> int:
        return self.join.out_channels

    def forward(self, input: Any, **kwargs: Any) -> torch.Tensor:
        deep_output = self.deep(input, **kwargs)
        shallow_output = self.shallow(input, **kwargs)
        return cast(torch.Tensor, self.join(deep_output, shallow_output))


def level(
    prev_level_block: Optional[SequentialWithOutChannels],
    impl_params: bool = True,
    instance_norm: bool = True,
    in_channels: int = 3,
    num_noise_channels: int = 3,
    inplace: bool = True,
) -> SequentialWithOutChannels:
    def conv_sequence(
        in_channels: int, out_channels: int, use_noise: bool = False
    ) -> SequentialWithOutChannels:
        modules: List[Tuple[str, nn.Module]] = []

        if use_noise:
            noise_module = noise(
                in_channels=in_channels, num_noise_channels=num_noise_channels,
            )
            in_channels = noise_module.out_channels
            modules.append(("noise", noise_module))

        conv_seq = ConvSequence(
            in_channels,
            out_channels,
            impl_params=impl_params,
            instance_norm=instance_norm,
            inplace=inplace,
        )

        if not use_noise:
            return conv_seq

        modules.append(("conv_seq", conv_seq))
        return SequentialWithOutChannels(OrderedDict(modules))

    use_noise = not impl_params
    shallow_branch = conv_sequence(in_channels, out_channels=8, use_noise=use_noise)

    if prev_level_block is None:
        return shallow_branch

    deep_branch = HourGlassBlock(prev_level_block)
    branch_block = BranchBlock(deep_branch, shallow_branch, instance_norm=instance_norm)

    output_conv_seq = conv_sequence(
        branch_block.out_channels, branch_block.out_channels
    )

    return SequentialWithOutChannels(
        OrderedDict((("branch", branch_block), ("output_conv_seq", output_conv_seq)))
    )


class Transformer(nn.Sequential):
    def __init__(
        self,
        levels: int,
        impl_params: bool = True,
        instance_norm: bool = True,
        init_weights: bool = True,
    ) -> None:
        pyramid = None
        for _ in range(levels):
            pyramid = level(
                pyramid, impl_params=impl_params, instance_norm=instance_norm,
            )

        if impl_params:
            output_conv = cast(
                Union[nn.Conv2d, ConvBlock],
                nn.Conv2d(
                    cast(int, cast(SequentialWithOutChannels, pyramid).out_channels),
                    3,
                    1,
                    1,
                ),
            )
        else:
            output_conv = cast(
                Union[nn.Conv2d, ConvBlock],
                ConvBlock(
                    cast(int, cast(SequentialWithOutChannels, pyramid).out_channels),
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                ),
            )

        super().__init__(
            OrderedDict(
                cast(
                    Tuple[Tuple[str, nn.Module]],
                    (("image_pyramid", pyramid), ("output_conv", output_conv)),
                )
            )
        )

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


def transformer(
    style: Optional[str] = None,
    impl_params: bool = True,
    instance_norm: bool = True,
    levels: int = 6,
) -> Transformer:
    return Transformer(levels, impl_params=impl_params, instance_norm=instance_norm)
