import itertools
from collections import OrderedDict

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

from pystiche.image import extract_image_size, extract_num_channels
from pystiche.misc import to_2d_arg
from pystiche_papers import utils


def test_Identity():
    input = torch.empty(1)
    model = utils.Identity()
    assert model(input) is input


@pytest.fixture(scope="module")
def double_module():
    class Double(nn.Module):
        def forward(self, x):
            return x * 2.0

    return Double()


def test_ResidualBlock(double_module):
    input = torch.tensor(1.0)
    model = utils.ResidualBlock(double_module)
    assert model(input) == ptu.approx(3.0)


def test_ResidualBlock_shortcut(double_module):
    input = torch.tensor(1.0)
    model = utils.ResidualBlock(
        double_module, shortcut=utils.ResidualBlock(double_module)
    )
    assert model(input) == ptu.approx(5.0)


def test_SequentialWithOutChannels(subtests):
    sequential_modules = (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))
    sequential_module_dict = OrderedDict(
        ((str(idx), module) for idx, module in enumerate(sequential_modules))
    )
    for out_channel_name, out_channels, args in (
        (None, 5, sequential_modules),
        (0, 3, sequential_modules),
        (1, 5, sequential_modules),
        ("0", 3, (sequential_module_dict,)),
        ("1", 5, (sequential_module_dict,)),
    ):
        with subtests.test(out_channel_name=out_channel_name):
            sequential = utils.SequentialWithOutChannels(
                *args, out_channel_name=out_channel_name
            )
            assert sequential.out_channels == out_channels


def test_SequentialWithOutChannels_forward_behaviour(input_image):
    sequential_modules = (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))
    sequential = utils.SequentialWithOutChannels(*sequential_modules)
    actual = sequential(input_image)
    desired = input_image
    for module in sequential_modules:
        desired = module(desired)
    ptu.assert_allclose(actual, desired)


@pytest.fixture
def same_size_conv_params():
    kernel_sizes = (3, 4, (3, 4), (4, 3))
    strides = (1, 2, (1, 2), (2, 1))
    dilations = (1, 2, (1, 2), (2, 1))
    return tuple(
        dict(kernel_size=kernel_size, stride=stride, dilation=dilation)
        for kernel_size, stride, dilation in itertools.product(
            kernel_sizes, strides, dilations
        )
    )


def test_SameSizeConv2d(subtests, same_size_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in same_size_conv_params:
        conv = utils.SameSizeConv2d(in_channels, out_channels, **params)
        output_image = conv(input_image)

        actual = extract_image_size(output_image)
        expected = tuple(
            side_length // stride
            for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
        )

        msg = (
            f"{', '.join((f'{key}={val}' for key, val in params.items()))}: "
            f"{actual} != {expected}"
        )
        assert actual == expected, msg


def test_SameSizeConvTranspose2d(subtests, same_size_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in same_size_conv_params:
        conv = utils.SameSizeConvTranspose2d(in_channels, out_channels, **params)
        output_image = conv(input_image)

        actual = extract_image_size(output_image)
        expected = tuple(
            side_length * stride
            for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
        )

        msg = (
            f"{', '.join((f'{key}={val}' for key, val in params.items()))}: "
            f"{actual} != {expected}"
        )
        assert actual == expected, msg


def test_SameSizeConv2d_padding():
    with pytest.raises(RuntimeError):
        utils.SameSizeConv2d(1, 1, 3, padding=1)


def test_SameSizeConv2d_repr_smoke():
    same_size_conv = utils.SameSizeConv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=1,
        stride=1,
        dilation=2,
        groups=2,
        bias=True,
        padding_mode="reflect",
    )
    assert isinstance(repr(same_size_conv), str)


def test_SameSizeConvTranspose2d_output_padding():
    with pytest.raises(RuntimeError):
        utils.SameSizeConvTranspose2d(1, 1, 3, output_padding=1)
