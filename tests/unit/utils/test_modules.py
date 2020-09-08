from collections import OrderedDict

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

from pystiche.image import extract_image_size, extract_num_channels
from pystiche.misc import to_2d_arg
from pystiche_papers import utils

from tests.utils import generate_param_combinations


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
    return tuple(
        generate_param_combinations(
            kernel_size=(3, 4, (3, 4), (4, 3)),
            stride=(1, 2, (1, 2), (2, 1)),
            dilation=(1, 2, (1, 2), (2, 1)),
        )
    )


def test_SameSizeConv2d(subtests, same_size_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in same_size_conv_params:
        with subtests.test(**params):
            conv = utils.SameSizeConv2d(in_channels, out_channels, **params)
            output_image = conv(input_image)

            actual = extract_image_size(output_image)
            expected = tuple(
                side_length // stride
                for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
            )

            assert actual == expected


def test_SameSizeConvTranspose2d(subtests, same_size_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in same_size_conv_params:
        with subtests.test(**params):
            conv = utils.SameSizeConvTranspose2d(in_channels, out_channels, **params)
            output_image = conv(input_image)

            actual = extract_image_size(output_image)
            expected = tuple(
                side_length * stride
                for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
            )
            assert actual == expected


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


def test_SameSizeConv2d_state_dict():
    kwargs = dict(in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv = nn.Conv2d(**kwargs)
    same_size_conv = utils.SameSizeConv2d(**kwargs)

    state_dict = conv.state_dict()
    same_size_conv.load_state_dict(state_dict)
    ptu.assert_allclose(same_size_conv.state_dict(), state_dict)


def test_SameSizeConvTranspose2d_output_padding():
    with pytest.raises(RuntimeError):
        utils.SameSizeConvTranspose2d(1, 1, 3, output_padding=1)


def test_SameSizeConvTranspose2d_state_dict():
    kwargs = dict(in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv = nn.ConvTranspose2d(**kwargs)
    same_size_conv = utils.SameSizeConvTranspose2d(**kwargs)

    state_dict = conv.state_dict()
    same_size_conv.load_state_dict(state_dict)
    ptu.assert_allclose(same_size_conv.state_dict(), state_dict)
