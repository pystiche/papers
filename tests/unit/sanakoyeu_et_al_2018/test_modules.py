import pytest

import torch
from torch import nn

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc
from pystiche_papers.utils import Identity, ResidualBlock, same_size_padding


def test_get_padding(subtests):
    kernel_size = 3
    for str_padding, desired in (
        ("same", same_size_padding(kernel_size)),
        ("valid", 0),
    ):
        with subtests.test(str_padding):
            actual = paper.get_padding(str_padding, kernel_size)
            assert actual == desired


def test_get_padding_wrong_string():
    kernel_size = 3
    with pytest.raises(ValueError):
        paper.get_padding("", kernel_size)


def test_get_activation(subtests):
    for str_act, desired in (("relu", nn.ReLU), ("lrelu", nn.LeakyReLU)):
        with subtests.test(str_act):
            actual = paper.activation(str_act)

            assert isinstance(actual, desired)

            with subtests.test("inplace"):
                assert actual.inplace

            if isinstance(actual, nn.LeakyReLU):
                with subtests.test("slope"):
                    assert actual.negative_slope == pytest.approx(0.2)


def test_get_activation_wrong_string():
    with pytest.raises(ValueError):
        paper.activation("")


def test_conv(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 2
    padding = "valid"
    conv = paper.conv(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding
    )

    assert isinstance(conv, nn.Conv2d)
    with subtests.test("in_channels"):
        assert conv.in_channels == in_channels
    with subtests.test("out_channels"):
        assert conv.out_channels == out_channels
    with subtests.test("kernel_size"):
        assert conv.kernel_size == misc.to_2d_arg(kernel_size)
    with subtests.test("stride"):
        assert conv.stride == misc.to_2d_arg(stride)
    with subtests.test("padding"):
        assert conv.padding == misc.to_2d_arg(0)


def test_ConvBlock(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1

    for act in ("relu", "lrelu", None):
        with subtests.test(act):
            conv_block = paper.ConvBlock(
                in_channels, out_channels, kernel_size, stride=stride, act=act
            )

            assert isinstance(conv_block, paper.ConvBlock)

            with subtests.test("modules"):
                assert len(conv_block) == 3 if act is not None else 2
                assert isinstance(conv_block[0], nn.Conv2d)
                assert isinstance(conv_block[1], nn.InstanceNorm2d)
                if act is not None:
                    assert isinstance(
                        conv_block[2], nn.ReLU if act == "relu" else nn.LeakyReLU
                    )


def test_ConvTransponseBlock(subtests, input_image):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 2

    conv_transponse_block = paper.ConvTransponseBlock(
        in_channels, out_channels, kernel_size, stride=stride
    )

    output_image = conv_transponse_block(input_image)
    assert isinstance(output_image, torch.Tensor)

    with subtests.test("conv_module"):
        assert isinstance(conv_transponse_block.conv, paper.ConvBlock)

    with subtests.test("interpolate"):
        desired_image = nn.functional.interpolate(
            input_image, scale_factor=stride, mode="nearest"
        )
        assert output_image.size() == desired_image.size()


def test_residual_block(subtests, input_image):
    channels = 3
    residual_block = paper.residual_block(channels)

    assert isinstance(residual_block, ResidualBlock)

    with subtests.test("residual"):
        assert isinstance(residual_block.residual, nn.Sequential)
        assert len(residual_block.residual) == 4

    with subtests.test("shortcut"):
        assert isinstance(residual_block.shortcut, Identity)

    with subtests.test("forward size"):
        output_image = residual_block(input_image)
        assert output_image.size() == input_image.size()
