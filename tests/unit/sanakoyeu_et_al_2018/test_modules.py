import pytest

import pytorch_testing_utils as ptu
from torch import nn

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc
from pystiche_papers.utils import AutoPadAvgPool2d, AutoPadConv2d, ResidualBlock


def test_get_activation(subtests):
    for str_act, desired in (("relu", nn.ReLU), ("lrelu", nn.LeakyReLU)):
        with subtests.test(str_act):
            actual = paper.get_activation(str_act)

            assert isinstance(actual, desired)

            with subtests.test("inplace"):
                assert actual.inplace

            if isinstance(actual, nn.LeakyReLU):
                with subtests.test("slope"):
                    assert actual.negative_slope == pytest.approx(0.2)


def test_conv(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 2
    for padding in (None, 0):
        with subtests.test(padding=padding):
            conv = paper.conv(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )

            if padding is None:
                assert isinstance(conv, AutoPadConv2d)
            else:
                assert isinstance(conv, nn.Conv2d)
                with subtests.test("padding"):
                    assert conv.padding == misc.to_2d_arg(padding)

            with subtests.test("in_channels"):
                assert conv.in_channels == in_channels
            with subtests.test("out_channels"):
                assert conv.out_channels == out_channels
            with subtests.test("kernel_size"):
                assert conv.kernel_size == misc.to_2d_arg(kernel_size)
            with subtests.test("stride"):
                assert conv.stride == misc.to_2d_arg(stride)


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


def test_UpsampleConvBlock(subtests, input_image):
    in_channels = out_channels = 3
    kernel_size = 3
    scale_factor = 2

    conv_block = paper.ConvBlock(in_channels, out_channels, kernel_size)
    upsample_conv_block = paper.UpsampleConvBlock(
        in_channels, out_channels, kernel_size, scale_factor=scale_factor
    )
    upsample_conv_block.load_state_dict(conv_block.state_dict())

    output = upsample_conv_block(input_image)
    ptu.assert_allclose(
        output,
        conv_block(
            nn.functional.interpolate(
                input_image, scale_factor=scale_factor, mode="nearest"
            )
        ),
    )


def test_residual_block(subtests, input_image):
    channels = 3

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            residual_block = paper.residual_block(channels, impl_params=impl_params)

            assert isinstance(residual_block, ResidualBlock)

            with subtests.test("residual"):
                residual = residual_block.residual
                assert isinstance(residual, nn.Sequential)
                assert len(residual) == 2

                if impl_params:
                    assert len(residual[0]) == 3
                    assert isinstance(residual[0][-1], nn.ReLU)

            with subtests.test("forward size"):
                output_image = residual_block(input_image)
                assert output_image.size() == input_image.size()


def test_discriminator_modules(subtests):
    channel_config = [
        (3, 128),
        (128, 128),
        (128, 256),
        (256, 512),
        (512, 512),
        (512, 1024),
        (1024, 1024),
    ]

    discriminator = paper.Discriminator()

    in_out_channels = []
    module_names = []
    for name, module in discriminator.named_children():
        with subtests.test("modules"):
            assert isinstance(module, paper.ConvBlock)
            in_out_channels.append((module[0].in_channels, module[0].out_channels))
            module_names.append(name)

    with subtests.test("channel_config"):
        assert in_out_channels == channel_config


def test_TransformerBlock(subtests, input_image):
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformer_block = paper.TransformerBlock(impl_params=impl_params)

            with subtests.test("module"):
                for module in transformer_block.children():
                    assert isinstance(
                        module, AutoPadAvgPool2d if impl_params else AutoPadConv2d
                    )

            with subtests.test("forward_size"):
                output_image = transformer_block(input_image)
                assert output_image.size() == input_image.size()


def test_prediction_module(subtests):
    in_channels = 3
    kernel_size = 3

    prediction_module = paper.prediction_module(in_channels, kernel_size)

    assert isinstance(prediction_module, nn.Conv2d)

    with subtests.test("in_channels"):
        assert prediction_module.in_channels == in_channels
    with subtests.test("out_channels"):
        assert prediction_module.out_channels == 1

    with subtests.test("kernel_size"):
        assert prediction_module.kernel_size == misc.to_2d_arg(kernel_size)

    with subtests.test("stride"):
        assert prediction_module.stride == misc.to_2d_arg(1)
