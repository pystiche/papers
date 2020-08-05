import pytest
from collections import OrderedDict

import pytorch_testing_utils as ptu
import torch
from torch import nn

from pystiche_papers import utils
from pystiche.image import extract_num_channels


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
    sequentialpaper = (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))
    sequential_module_dict = OrderedDict(
        ((str(idx), module) for idx, module in enumerate(sequentialpaper))
    )
    for out_channel_name, out_channels, args in (
        (None, 5, sequentialpaper),
        (0, 3, sequentialpaper),
        (1, 5, sequentialpaper),
        ("0", 3, (sequential_module_dict,)),
        ("1", 5, (sequential_module_dict,)),
    ):
        with subtests.test(out_channel_name=out_channel_name):
            sequential = utils.SequentialWithOutChannels(
                *args, out_channel_name=out_channel_name
            )
            assert sequential.out_channels == out_channels


def test_AddNoiseChannels(subtests, input_image):
    in_channels = extract_num_channels(input_image)
    num_noise_channels = in_channels + 1
    module = utils.AddNoiseChannels(in_channels, num_noise_channels=num_noise_channels)

    assert isinstance(module, nn.Module)

    with subtests.test("in_channels"):
        assert module.in_channels == in_channels

    desired_out_channels = in_channels + num_noise_channels

    with subtests.test("out_channels"):
        assert module.out_channels == desired_out_channels

    with subtests.test("forward"):
        output_image = module(input_image)
        assert extract_num_channels(output_image) == desired_out_channels


def test_HourGlassBlock(subtests):
    downsample = nn.AvgPool2d(3, stride=2, padding=0)
    intermediate = nn.Conv2d(3, 3, 1)
    upsample = nn.Upsample(scale_factor=2.0)
    hour_glass = utils.HourGlassBlock(downsample, intermediate, upsample)

    assert isinstance(hour_glass, utils.HourGlassBlock)

    with subtests.test("down"):
        assert isinstance(hour_glass.down, nn.AvgPool2d)
    with subtests.test("intermediate"):
        assert hour_glass.intermediate is intermediate
    with subtests.test("up"):
        assert isinstance(hour_glass.up, nn.Upsample)
