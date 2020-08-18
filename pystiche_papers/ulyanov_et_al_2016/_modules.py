from collections import OrderedDict
from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn

from pystiche import meta as meta_
from pystiche import misc

from ..utils import SequentialWithOutChannels, is_valid_padding, same_size_padding


def join_channelwise(*inputs: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    return torch.cat(inputs, dim=channel_dim)


class AddNoiseChannels(nn.Module):
    r"""Adds noise channels in the size of the input to the input.

    Args:
        in_channels: Number of input channels.
        num_noise_channels: Number of additional noise channels. Defaults to ``3``.
    """

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
    r"""HourGlassBlock from :cite:`ULVL2016`.

    Args:
        intermediate: Middle :class:`~torch.nn.Module` of the block.
    """

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
    # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/models/pyramid.lua#L5
    return (
        nn.ReLU(inplace=inplace)
        if impl_params and instance_norm
        else nn.LeakyReLU(negative_slope=0.01, inplace=inplace)
    )


class ConvBlock(SequentialWithOutChannels):
    r"""ConvBlock from :cite:`ULVL2016`.

    Args:
        in_channels: Number of channels in the input.
        out_channels:  Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        stride: Stride of the convolution. Defaults to ``1``.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
        inplace: Can optionally do the operation in-place. Defaults to ``True``.

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
    r"""Sequence of convolutional blocks that occurs repeatedly in :cite:`ULVL2016`.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
        inplace: Can optionally do the operation in-place. Defaults to ``True``.

    """

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
    r"""JoinBlock from :cite:`ULVL2016`.

    Args:
        branch_in_channels: Number of channels in the branch input.
        names: Optional names for the blocks. If omitted, the blocks are numbered.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
        channel_dim: The dimension over which the tensors are concatenated. Defaults to
            ``1``.

    """

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
    r"""BranchBlock from :cite:`ULVL2016`.

    Args:
        deep_branch: Input from the branch one step deeper in the pyramid.
        shallow_branch: Input from the current branch.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
    """

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
    r"""Defines one level of the Transformer from :cite:`ULVL2016`.

    Args:
        prev_level_block: Optional Input from the previous level. If ``None``, only one
            ConvSequence is returned.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
        in_channels: Number of channels in the input image. Defaults to ``3``.
        num_noise_channels: Number of additional noise channels. Defaults to ``3``.
        inplace: Can optionally do the operation in-place. Defaults to ``True``.

    """

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
            # Just a torch.nn.Conv2d is used instead of the ConvBlock as described in
            # the paper.
            # https://github.com/pmeier/texture_nets/blob/aad2cc6f8a998fedc77b64bdcfe1e2884aa0fb3e/models/pyramid.lua#L61
            # https://github.com/pmeier/texture_nets/blob/b2097eccaec699039038970b191780f97c238816/models/pyramid.lua#L62
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
    r"""Transformer from :cite:`ULVL2016`.

    Args:
        style: FIXME this should be removed.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see FIXME.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between the github branches. For details see
            FIXME.
        levels: Number of levels in the Transformer. Defaults to ``6``.

    """
    return Transformer(levels, impl_params=impl_params, instance_norm=instance_norm)
