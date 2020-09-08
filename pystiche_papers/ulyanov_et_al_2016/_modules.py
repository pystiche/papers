from collections import OrderedDict
from math import sqrt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import torch
from torch import nn

from pystiche import meta as meta_
from pystiche import misc

from ..utils import AutoPadConv2d, SequentialWithOutChannels


def join_channelwise(*inputs: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
    return torch.cat(inputs, dim=channel_dim)


class AddNoiseChannels(nn.Module):
    r"""Adds white noise channels to the input.

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
    r"""Downsample the input to half the size using an :class:`~torch.nn.AvgPool2d`.

    Args:
        kernel_size: Size of the kernel. Defaults to ``2``.
        stride: Stride of the kernel. Defaults to ``2``.
        padding: Padding to be added on both sides. Defaults to ``0``.

    """
    return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)


def upsample() -> nn.Upsample:
    r"""Upsample the input to twice the size using an :class:`~torch.nn.Upsample`."""
    return nn.Upsample(scale_factor=2.0, mode="nearest")


class HourGlassBlock(SequentialWithOutChannels):
    r"""HourGlassBlock from :cite:`ULVL2016`.

    This block embeds an ``intermediate`` module between a :func:`downsample` and
    :func:`upsample` operation.

    Args:
        intermediate: Module in between the down- and upsampling.

    Attributes:
        out_channels: ``ìntermediate.out_channels``

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

    This block comprises a convolution followed by normalization and activation. The
    input is reflection-padded to preserve the size.

    Args:
        in_channels: Number of channels in the input.
        out_channels:  Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        stride: Stride of the convolution. Defaults to ``1``.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        inplace: If ``True`` perform the activation in-place.

    If ``impl_params and instance_norm is True`` the activation function is a
    :class:`~torch.nn.ReLU` otherwise a :class:`~torch.nn.LeakyReLU` with
    ``slope=0.01``.

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
        modules = (
            (
                "conv",
                AutoPadConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding_mode="reflect",
                ),
            ),
            ("norm", norm(out_channels, instance_norm)),
            ("act", activation(impl_params, instance_norm, inplace=inplace)),
        )
        super().__init__(OrderedDict(modules), out_channel_name="conv")


class ConvSequence(SequentialWithOutChannels):
    r"""Sequence of convolutional blocks that occurs repeatedly in :cite:`ULVL2016`.

    Each sequence contains three
    :class:`~pystiche_paper.ulyanov_et_al_2016._modules.ConvBlock` s. The
    first two use ``kernel_size == 3`` and the third one uses ``kernel_size == 1``.

    Args:
        in_channels: Number of channels in the input.
        out_channels: Number of channels produced by the convolution
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        inplace: If ``True`` perform the activation in-place.

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
    r"""JoinBlock from :cite:`ULVL2016` without upsampling.

    This block concatenates an arbitrary number of inputs along the ``channel _dim``
    with prefixed normalization modules.

    Args:
        branch_in_channels: Number of channels in the branch input.
        names: Optional names for the normalization modules. If omitted, the modules
            will be numbered.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
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

    Joins the branch from the previous pyramid level with the current using a
    :class:`~pystiche_paper.ulyanov_et_al_2016._modules.JoinBlock`.

    Args:
        deep_branch: Previous pyramid level.
        shallow_branch: Current pyramid level.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
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
    r"""Defines one level of the transformer from :cite:`ULVL2016`.

    The basic building block of a level is a :class:`ConvSequence` . If a previous
    level exists, i. e. the current level is not the first one, the previous level is
    incorporated by embedding it in an :class:`HourGlassBlock`. The outputs of both
    levels are joined with a :class:`BranchBlock` and finally fed through another
    :class:`ConvSequence`.

    Args:
        prev_level_block: Previous pyramid level. If given, it is incorporated in the
            current level.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        in_channels: Number of channels in the input image. Defaults to ``3``.
        num_noise_channels: Number of additional noise channels. Defaults to ``3``.
        inplace: If ``True`` perform the activation in-place.

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
        style: Style the transformer was trained on. Can be one of styles given by
            :func:`~pystiche_papers.ulyanov_et_al_2016.images`. If omitted, the
            transformer is initialized with random weights.
        impl_params: If ``True``, use the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper. For details see
            :ref:`here <table-hyperparameters-ulyanov_et_al_2016>`.
        instance_norm: If ``True``, use :class:`~torch.nn.InstanceNorm2d` rather than
            :class:`~torch.nn.BatchNorm2d` as described in the paper. Additionally this
            flag is used for switching between two reference implementations. For
            details see :ref:`here <table-branches-ulyanov_et_al_2016>`.
        levels: Number of levels in the transformer. Defaults to ``6``.

    """
    return Transformer(levels, impl_params=impl_params, instance_norm=instance_norm)
