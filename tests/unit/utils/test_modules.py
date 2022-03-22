from collections import OrderedDict

import pytest

from tests.utils import generate_param_combinations

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
def auto_pad_conv_params():
    return tuple(
        generate_param_combinations(
            kernel_size=(3, 4, (3, 4), (4, 3)),
            stride=(1, 2, (1, 2), (2, 1)),
            dilation=(1, 2, (1, 2), (2, 1)),
        )
    )


def test_AutoPadConv2d(subtests, auto_pad_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in auto_pad_conv_params:
        with subtests.test(**params):
            conv = utils.AutoPadConv2d(in_channels, out_channels, **params)
            output_image = conv(input_image)

            actual = extract_image_size(output_image)
            expected = tuple(
                side_length // stride
                for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
            )

            assert actual == expected


def test_AutoPadConvTranspose2d(subtests, auto_pad_conv_params, input_image):
    in_channels = out_channels = extract_num_channels(input_image)
    image_size = extract_image_size(input_image)

    for params in auto_pad_conv_params:
        with subtests.test(**params):
            conv = utils.AutoPadConvTranspose2d(in_channels, out_channels, **params)
            output_image = conv(input_image)

            actual = extract_image_size(output_image)
            expected = tuple(
                side_length * stride
                for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
            )
            assert actual == expected


def test_AutoPadConv2d_padding():
    with pytest.raises(RuntimeError):
        utils.AutoPadConv2d(1, 1, 3, padding=1)


def test_AutoPadConv2d_repr_smoke():
    auto_pad_conv = utils.AutoPadConv2d(
        in_channels=2,
        out_channels=2,
        kernel_size=1,
        stride=1,
        dilation=2,
        groups=2,
        bias=True,
        padding_mode="reflect",
    )
    assert isinstance(repr(auto_pad_conv), str)


def test_AutoPadConv2d_state_dict():
    kwargs = dict(in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv = nn.Conv2d(**kwargs)
    auto_pad_conv = utils.AutoPadConv2d(**kwargs)

    state_dict = conv.state_dict()
    auto_pad_conv.load_state_dict(state_dict)
    ptu.assert_allclose(auto_pad_conv.state_dict(), state_dict)


def test_AutoPadConvTranspose2d_output_padding():
    with pytest.raises(RuntimeError):
        utils.AutoPadConvTranspose2d(1, 1, 3, output_padding=1)


def test_AutoPadConvTranspose2d_state_dict():
    kwargs = dict(in_channels=1, out_channels=2, kernel_size=3, bias=True)
    conv = nn.ConvTranspose2d(**kwargs)
    auto_pad_conv = utils.AutoPadConvTranspose2d(**kwargs)

    state_dict = conv.state_dict()
    auto_pad_conv.load_state_dict(state_dict)
    ptu.assert_allclose(auto_pad_conv.state_dict(), state_dict)


@pytest.fixture
def auto_pad_pool_params():
    return tuple(
        generate_param_combinations(
            kernel_size=(3, 4, (3, 4), (4, 3)),
            stride=(1, 2, (1, 2), (2, 1)),
        )
    )


def test_AutoPadAvgPool2d(subtests, auto_pad_pool_params, input_image):
    image_size = extract_image_size(input_image)

    for params in auto_pad_pool_params:
        with subtests.test(**params):
            conv = utils.AutoPadAvgPool2d(**params)
            output_image = conv(input_image)

            actual = extract_image_size(output_image)
            expected = tuple(
                side_length // stride
                for side_length, stride in zip(image_size, to_2d_arg(params["stride"]))
            )

            assert actual == expected


def test_AutoPadAvgPool2d_count_include_pad(input_image):
    kernel_size = 5

    manual = nn.AvgPool2d(
        kernel_size, stride=1, padding=(kernel_size - 1) // 2, count_include_pad=False
    )
    auto = utils.AutoPadAvgPool2d(kernel_size, stride=1, count_include_pad=False)

    ptu.assert_allclose(auto(input_image), manual(input_image), rtol=1e-6)


def test_AutoPadAvgPool2d_count_include_pad_stride(input_image):
    with pytest.raises(RuntimeError):
        utils.AutoPadAvgPool2d(kernel_size=2, stride=2, count_include_pad=False)


def test_AutoPadAvgPool1d_count_include_pad():
    kernel_size = 5

    from pystiche.misc import to_1d_arg
    from pystiche_papers.utils.modules import _AutoPadAvgPoolNdMixin

    class AutoPadAvgPool1d(_AutoPadAvgPoolNdMixin, nn.AvgPool1d):
        def __init__(
            self,
            kernel_size,
            stride=None,
            **kwargs,
        ) -> None:
            kernel_size = to_1d_arg(kernel_size)
            stride = kernel_size if stride is None else to_1d_arg(stride)
            super().__init__(kernel_size, stride=stride, **kwargs)

    torch.manual_seed(0)
    input = torch.rand(1, 1, 32)

    manual = nn.AvgPool1d(
        kernel_size, stride=1, padding=(kernel_size - 1) // 2, count_include_pad=False
    )
    auto = AutoPadAvgPool1d(kernel_size, stride=1, count_include_pad=False)

    ptu.assert_allclose(auto(input), manual(input), rtol=1e-6)


def test_AutoPadAvgPool3d_count_include_pad(input_image):
    from pystiche.misc import to_3d_arg
    from pystiche_papers.utils.modules import _AutoPadAvgPoolNdMixin

    class AutoPadAvgPool3d(_AutoPadAvgPoolNdMixin, nn.AvgPool3d):
        def __init__(
            self,
            kernel_size,
            stride=None,
            **kwargs,
        ) -> None:
            kernel_size = to_3d_arg(kernel_size)
            stride = kernel_size if stride is None else to_3d_arg(stride)
            super().__init__(kernel_size, stride=stride, **kwargs)

    with pytest.raises(RuntimeError):
        AutoPadAvgPool3d(kernel_size=2, count_include_pad=False)


def test_AutoPadAvgPool2d_repr_smoke():
    auto_pad_conv = utils.AutoPadAvgPool2d(
        kernel_size=1,
        stride=1,
    )
    assert isinstance(repr(auto_pad_conv), str)
