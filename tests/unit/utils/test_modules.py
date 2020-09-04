from collections import OrderedDict
import itertools
import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

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


def test_PaddedConv2D(subtests, input_image):
    in_channels = out_channels = 3

    for kernel_size, padding in itertools.product((3, 4, (3, 3), (3, 4), (4, 3), (4, 4)), ("same", "valid", "full")):
        with subtests.test(kernel_size=kernel_size, padding=padding):
            padded_conv = utils.PaddedConv2D(in_channels, out_channels, kernel_size, padding=padding)

            with subtests.test("padding"):
                assert padded_conv.padding == to_2d_arg(utils.get_padding(padding, kernel_size))

            with subtests.test("forward_size"):
                output_image = padded_conv(input_image)
                if padding == "same":
                    desired_image_size = input_image.size()
                else:
                    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=utils.get_padding(padding, kernel_size))
                    desired_image_size = conv(input_image).size()

                assert desired_image_size == output_image.size()

