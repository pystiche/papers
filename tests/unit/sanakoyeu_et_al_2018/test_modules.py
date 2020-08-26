import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc
from pystiche.enc import SequentialEncoder
from pystiche_papers.utils import ResidualBlock, same_size_padding


def test_get_padding(subtests):
    kernel_size = 3
    for str_padding, desired in (
        ("same", same_size_padding(kernel_size)),
        ("valid", 0),
    ):
        with subtests.test(str_padding):
            actual = paper.get_padding(str_padding, kernel_size)
            assert actual == desired


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
        assert len(residual_block.residual) == 4

    with subtests.test("forward size"):
        output_image = residual_block(input_image)
        assert output_image.size() == input_image.size()


def test_transformer_encoder(subtests):
    channel_config = [(3, 32), (32, 32), (32, 64), (64, 128), (128, 256)]

    encoder = paper.transformer_encoder()

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


def test_transformer_decoder(subtests):
    num_res_block = 2
    channel_config = [
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 128),
        (128, 64),
        (64, 32),
        (32, 3),
    ]

    decoder = paper.transformer_decoder(num_res_block=num_res_block)

    assert isinstance(decoder, pystiche.SequentialModule)

    in_out_channels = []
    for i, module in enumerate(decoder.children()):
        with subtests.test("modules"):
            if i in range(0, num_res_block):
                with subtests.test("residualblocks"):
                    assert isinstance(module, ResidualBlock)
                    in_out_channels.append(
                        (
                            module.residual[1][0].in_channels,
                            module.residual[-1][0].out_channels,
                        )
                    )
            if i in range(num_res_block, num_res_block + 4):
                assert isinstance(module, paper.UpsampleConvBlock)
                in_out_channels.append(
                    (module.conv[0].in_channels, module.conv[0].out_channels)
                )
            if i == num_res_block + 4:
                with subtests.test("padding_module"):
                    assert isinstance(module, nn.ReflectionPad2d)

            if i == num_res_block + 5:
                with subtests.test("last_conv"):
                    assert isinstance(module, nn.Conv2d)
                    with subtests.test("kernel_size"):
                        assert module.kernel_size == misc.to_2d_arg(7)
                    with subtests.test("stride"):
                        assert module.stride == misc.to_2d_arg(1)
                    with subtests.test("padding"):
                        assert module.padding == misc.to_2d_arg(0)
                in_out_channels.append((module.in_channels, module.out_channels))

    with subtests.test("channel_config"):
        assert in_out_channels == channel_config


def test_DecoderSigmoidOutput(input_image):
    image = input_image * 100
    decoder_output_module = paper.DecoderSigmoidOutput()
    actual = decoder_output_module(image)

    assert isinstance(actual, torch.Tensor)
    desired = torch.sigmoid(image) * 2 - 1
    ptu.assert_allclose(actual, desired)


def test_Decoder(subtests):
    decoder = paper.Decoder()

    with subtests.test("decoder"):
        assert isinstance(decoder.decoder, pystiche.SequentialModule)

    with subtests.test("output_module"):
        assert isinstance(decoder.output_module, paper.DecoderSigmoidOutput)


def test_Transformer_smoke(subtests, input_image):
    transformer = paper.Transformer()
    output_image = transformer(input_image)

    with subtests.test("encoder"):
        assert isinstance(transformer.encoder, SequentialEncoder)

    with subtests.test("decoder"):
        assert isinstance(transformer.decoder, paper.Decoder)

    with subtests.test("forward size"):
        assert input_image.size() == output_image.size()
