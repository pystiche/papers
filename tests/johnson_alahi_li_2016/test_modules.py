import itertools

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche
from pystiche.misc import to_2d_arg
from pystiche_papers.johnson_alahi_li_2016 import modules
from pystiche_papers.utils import ResidualBlock

from .._utils import should_be_available


def test_select_url(subtests):

    available_configs = (
        ("candy", True, True),
        ("composition_vii", True, False),
        ("feathers", True, True),
        ("la_muse", True, False),
        ("la_muse", True, True),
        ("mosaic", True, True),
        ("starry_night", True, False),
        ("the_scream", True, True),
        ("the_wave", True, False),
        ("udnie", True, True),
    )
    styles = (
        "candy",
        "composition_vii",
        "feathers",
        "la_muse",
        "mosaic",
        "starry_night",
        "the_scream",
        "the_wave",
        "udnie",
    )
    instance_norm_configs = (True, False)
    impl_params_configs = (True, False)

    for style, impl_params, instance_norm in itertools.product(
        styles, instance_norm_configs, impl_params_configs
    ):
        with subtests.test(style):
            case = (style, impl_params, instance_norm)
            if should_be_available(case, available_configs):
                url = modules.select_url(
                    style,
                    weights="author",
                    impl_params=impl_params,
                    instance_norm=instance_norm,
                )
                assert True
                # assert_is_downloadable(url) # TODO: use this here
            else:
                with pytest.raises(RuntimeError):
                    modules.select_url(
                        style,
                        weights="author",
                        impl_params=impl_params,
                        instance_norm=instance_norm,
                    )


def test_get_conv(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    padding = (None, 1, (1, 1))
    upsample = (True, False)
    configs = itertools.product(padding, upsample)
    for padding, upsample in configs:
        conv_module = modules.get_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            upsample=upsample,
        )

        with subtests.test("module"):
            assert isinstance(
                type(conv_module),
                type(nn.ConvTranspose2d) if upsample else type(nn.Conv2d),
            )

        with subtests.test("in_channels"):
            assert conv_module.in_channels == in_channels

        with subtests.test("out_channels"):
            assert conv_module.out_channels == out_channels

        with subtests.test("kernel_size"):
            assert conv_module.kernel_size == to_2d_arg(kernel_size)

        with subtests.test("stride"):
            assert conv_module.stride == to_2d_arg(stride)

        with subtests.test("padding"):
            assert conv_module.padding == (1, 1)

        if upsample:
            with subtests.test("output_padding"):
                assert conv_module.output_padding == to_2d_arg(0)


def test_get_norm(subtests):
    out_channels = 3
    for instance_norm in (True, False):
        with subtests.test(instance_norm=instance_norm):
            norm_module = modules.get_norm(out_channels, instance_norm=instance_norm)

            with subtests.test("module"):
                assert isinstance(
                    type(norm_module),
                    type(nn.InstanceNorm2d) if instance_norm else type(nn.BatchNorm2d),
                )

            with subtests.test("out_channels"):
                assert norm_module.num_features == out_channels

            with subtests.test("eps"):
                assert norm_module.eps == pytest.approx(1e-5)

            with subtests.test("momentum"):
                assert norm_module.momentum == pytest.approx(1e-1)

            with subtests.test("affine"):
                assert norm_module.affine

            with subtests.test("track_running_stats"):
                assert norm_module.track_running_stats


def test_johnson_alahi_li_2016_conv_block(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    for relu in (True, False):
        with subtests.test(relu=relu):
            conv_block = modules.johnson_alahi_li_2016_conv_block(
                in_channels, out_channels, kernel_size, stride=stride, relu=relu
            )

            with subtests.test("block"):
                assert isinstance(type(conv_block), type(nn.Sequential))

            assert len(conv_block) == 3 if relu else 2
            assert isinstance(type(conv_block[0]), type(nn.Conv2d))
            assert isinstance(type(conv_block[1]), type(nn.InstanceNorm2d))
            if relu:
                assert isinstance(type(conv_block[2]), type(nn.ReLU))
                assert conv_block[2].inplace


def test_johnson_alahi_li_2016_residual_block(subtests, input_image):
    channels = 3
    residual_block = modules.johnson_alahi_li_2016_residual_block(channels)

    with subtests.test("block"):
        assert isinstance(type(residual_block), type(ResidualBlock))

    with subtests.test("residual"):
        assert isinstance(type(residual_block.residual), type(nn.Sequential))
        for i in range(0, 2):
            assert isinstance(type(residual_block.residual[i]), type(nn.Sequential))

    with subtests.test("shortcut"):
        assert isinstance(type(residual_block.shortcut), type(nn.Module))
        ptu.assert_allclose(
            input_image[:, :, 2:-2, 2:-2], residual_block.shortcut(input_image)
        )


def test_johnson_alahi_li_2016_transformer_encoder(subtests):
    channel_configs = [
        [(3, 16), (16, 32), (32, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)],
        [
            (3, 32),
            (32, 64),
            (64, 128),
            (128, 128),
            (128, 128),
            (128, 128),
            (128, 128),
            (128, 128),
        ],
    ]

    for instance_norm, channel_config in zip((True, False), channel_configs):

        encoder = modules.johnson_alahi_li_2016_transformer_encoder(
            instance_norm=instance_norm
        )

        with subtests.test("encoder"):
            assert isinstance(type(encoder), type(pystiche.SequentialModule))

        in_out_channels = []
        for i, module in enumerate(encoder.children()):
            if i == 0:
                with subtests.test("padding_module"):
                    assert isinstance(type(module), type(nn.ReflectionPad2d))

            if i in range(1, 4):
                with subtests.test("conv_layer"):
                    assert isinstance(type(module), type(nn.Sequential))
                    in_out_channels.append(
                        (module[0].in_channels, module[0].out_channels)
                    )
            if i in range(4, 9):
                with subtests.test("residualblocks"):
                    assert isinstance(type(module), type(ResidualBlock))
                    in_out_channels.append(
                        (
                            module.residual[0][0].in_channels,
                            module.residual[-1][0].out_channels,
                        )
                    )

        with subtests.test("channel_config"):
            assert in_out_channels == channel_config


def test_johnson_alahi_li_2016_transformer_decoder(subtests):
    channel_configs = [[(64, 32), (32, 16), (16, 3)], [(128, 64), (64, 32), (32, 3)]]

    for instance_norm, channel_config in zip((True, False), channel_configs):
        with subtests.test(instance_norm=instance_norm):
            decoder = modules.johnson_alahi_li_2016_transformer_decoder(
                instance_norm=instance_norm
            )

            with subtests.test("decoder"):
                assert isinstance(type(decoder), type(pystiche.SequentialModule))

            in_out_channels = []
            for i, module in enumerate(decoder.children()):
                if i in range(2):
                    with subtests.test("conv_layer"):
                        assert isinstance(type(module), type(nn.Sequential))
                        in_out_channels.append(
                            (module[0].in_channels, module[0].out_channels)
                        )
                if i == 2:
                    with subtests.test("output_conv"):
                        assert isinstance(type(module), type(nn.Conv2d))
                        in_out_channels.append(
                            (module.in_channels, module.out_channels)
                        )

            with subtests.test("channel_config"):
                assert in_out_channels == channel_config


def test_johnson_alahi_li_2016_transformer_decoder_value_range_delimiter(
    subtests, input_image
):
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            decoder = modules.johnson_alahi_li_2016_transformer_decoder(
                impl_params=impl_params
            )

            module = [x for x in decoder.children()][-1]
            assert isinstance(type(module), type(nn.Module))

            with subtests.test("delimiter"):
                actual = module(input_image)
                desired = (
                    150.0 * torch.tanh(input_image)
                    if impl_params
                    else torch.sigmoid(2.0 * input_image)
                )

                ptu.assert_allclose(actual, desired)


def test_JohnsonAlahiLi2016Transformer(image_medium):
    transformer = modules.JohnsonAlahiLi2016Transformer()

    assert isinstance(type(transformer.encoder), type(pystiche.SequentialModule))
    assert isinstance(type(transformer.decoder), type(pystiche.SequentialModule))

    output_image = transformer(image_medium)
    assert image_medium.size() == output_image.size()


def test_johnson_alahi_li_2016_transformer():
    transformer = modules.johnson_alahi_li_2016_transformer()

    assert isinstance(type(transformer), type(modules.JohnsonAlahiLi2016Transformer))


def test_johnson_alahi_li_2016_transformer_wrong_impl_params_instance_norm():
    with pytest.raises(RuntimeError):
        modules.johnson_alahi_li_2016_transformer(instance_norm=True, impl_params=False)
