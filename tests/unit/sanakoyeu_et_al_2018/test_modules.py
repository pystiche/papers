import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc
from pystiche.enc import SequentialEncoder
from pystiche_papers.utils import AutoPadConv2d, ResidualBlock


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

    upsample_conv_block = paper.UpsampleConvBlock(
        in_channels, out_channels, kernel_size, scale_factor=scale_factor
    )

    output_image = upsample_conv_block(input_image)
    assert isinstance(output_image, torch.Tensor)

    with subtests.test("conv_module"):
        assert isinstance(upsample_conv_block.conv, paper.ConvBlock)

    with subtests.test("interpolate"):
        desired_image = nn.functional.interpolate(
            input_image, scale_factor=scale_factor, mode="nearest"
        )
        assert output_image.size() == desired_image.size()


def test_residual_block(subtests, input_image):
    channels = 3
    residual_block = paper.residual_block(channels)

    assert isinstance(residual_block, ResidualBlock)

    with subtests.test("residual"):
        assert isinstance(residual_block.residual, nn.Sequential)
        assert len(residual_block.residual) == 2

    with subtests.test("forward size"):
        output_image = residual_block(input_image)
        assert output_image.size() == input_image.size()


def test_encoder(subtests):
    channel_config = [(3, 32), (32, 32), (32, 64), (64, 128), (128, 256)]

    encoder = paper.encoder()

    assert isinstance(encoder, SequentialEncoder)

    in_out_channels = []
    for i, module in enumerate(encoder.children()):
        with subtests.test("modules"):
            if i == 0:
                with subtests.test("padding_module"):
                    assert isinstance(module, nn.ReflectionPad2d)
            else:
                assert isinstance(module, paper.ConvBlock)
                in_out_channels.append((module[0].in_channels, module[0].out_channels))

    with subtests.test("channel_config"):
        assert in_out_channels == channel_config


def test_decoder(subtests, input_image):
    num_residual_blocks = 2
    channel_config = [
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 3),
    ]

    decoder = paper.decoder(num_residual_blocks=num_residual_blocks)

    assert isinstance(decoder, pystiche.SequentialModule)

    in_out_channels = []
    children = decoder.children()
    with subtests.test("residual_blocks"):
        for _ in range(num_residual_blocks):
            module = next(children)
            assert isinstance(module, ResidualBlock)
            in_out_channels.append(
                (
                    module.residual[1][0].in_channels,
                    module.residual[-1][0].out_channels,
                )
            )

    with subtests.test("upsample_conv_blocks"):
        for _ in range(4):
            module = next(children)
            assert isinstance(module, paper.UpsampleConvBlock)
            in_out_channels.append(
                (module.conv[0].in_channels, module.conv[0].out_channels)
            )

    module = next(children)
    with subtests.test("last_conv"):
        assert isinstance(module, AutoPadConv2d)
        with subtests.test("kernel_size"):
            assert module.kernel_size == misc.to_2d_arg(7)
        with subtests.test("stride"):
            assert module.stride == misc.to_2d_arg(1)
        in_out_channels.append((module.in_channels, module.out_channels))

    module = next(children)
    with subtests.test("value_range_delimiter"):
        torch.manual_seed(0)
        input = torch.randn(10, 10)
        ptu.assert_allclose(module(input), torch.tanh(input / 2))

    with subtests.test("channel_config"):
        assert in_out_channels == channel_config


def test_Transformer_smoke(subtests, input_image):
    transformer = paper.Transformer()
    output_image = transformer(input_image)

    with subtests.test("encoder"):
        assert isinstance(transformer.encoder, SequentialEncoder)

    with subtests.test("decoder"):
        assert isinstance(transformer.decoder, pystiche.SequentialModule)

    with subtests.test("forward size"):
        assert input_image.size() == output_image.size()


def test_transformer():
    transformer = paper.transformer()
    assert isinstance(transformer, paper.Transformer)


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


def test_get_transformation_block(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 2
    padding = 2
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformation_block = paper.get_transformation_block(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                impl_params=impl_params,
            )

            assert isinstance(
                transformation_block, nn.AvgPool2d if impl_params else nn.Conv2d
            )

            with subtests.test("kernel_size"):
                assert (
                    transformation_block.kernel_size == misc.to_2d_arg(kernel_size)
                    if not impl_params
                    else kernel_size
                )
            with subtests.test("stride"):
                assert (
                    transformation_block.stride == misc.to_2d_arg(stride)
                    if not impl_params
                    else stride
                )
            with subtests.test("padding"):
                assert (
                    transformation_block.padding == misc.to_2d_arg(padding)
                    if not impl_params
                    else padding
                )

            if isinstance(transformation_block, nn.Conv2d):
                with subtests.test("in_channels"):
                    assert transformation_block.in_channels == in_channels
                with subtests.test("out_channels"):
                    assert transformation_block.out_channels == out_channels


def test_TransformerBlock(subtests, input_image):
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformer_block = paper.TransformerBlock(impl_params=impl_params)

            with subtests.test("module"):
                for module in transformer_block.children():
                    assert isinstance(
                        module, nn.AvgPool2d if impl_params else nn.Conv2d
                    )

            with subtests.test("forward_size"):
                output_image = transformer_block(input_image)
                assert output_image.size() == input_image.size()
