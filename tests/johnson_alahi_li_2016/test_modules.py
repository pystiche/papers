import itertools

import pytest

from torch import nn

from pystiche_papers.johnson_alahi_li_2016 import modules
from pystiche_papers.utils import ResidualBlock


def test_get_conv(subtests, mocker):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    padding = (None, 1, (1, 1))
    upsample = (True, False)
    configs = itertools.product(padding, upsample)
    for padding, upsample in configs:
        with subtests.test(upsample):
            mock = mocker.patch(
                "torch.nn.Conv2d" if upsample else "torch.nn.ConvTranspose2d"
            )

            modules.get_conv(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                upsample=upsample,
            )

            for call in mock.call_args_list:
                args, kwargs = call
                conv_in_channels, conv_out_channels, conv_kernel_size, = args
                conv_stride = kwargs["stride"]
                conv_padding = kwargs["padding"]

                with subtests.test("in_channels"):
                    assert conv_in_channels == in_channels

                with subtests.test("out_channels"):
                    assert conv_out_channels == out_channels

                with subtests.test("kernel_size"):
                    assert conv_kernel_size == kernel_size

                with subtests.test("stride"):
                    assert conv_stride == stride

                with subtests.test("padding"):
                    assert conv_padding == (1, 1)

                if upsample:
                    conv_output_padding = kwargs["output_padding"]

                    with subtests.test("output_padding"):
                        assert conv_output_padding == (1, 1)


def test_get_norm(subtests, mocker):
    out_channels = 3
    for instance_norm in (True, False):
        with subtests.test(instance_norm=instance_norm):
            mock = mocker.patch(
                "torch.nn.BatchNorm2d" if instance_norm else "torch.nn.InstanceNorm2d"
            )

            modules.get_norm(out_channels, instance_norm=instance_norm)

            for call in mock.call_args_list:
                args, kwargs = call
                norm_out_channels = args
                norm_eps = kwargs["eps"]
                norm_momentum = kwargs["momentum"]
                norm_affine = kwargs["affine"]
                norm_track_running_stats = kwargs["track_running_stats"]

                with subtests.test("out_channels"):
                    assert norm_out_channels == out_channels

                with subtests.test("eps"):
                    assert norm_eps == pytest.approx(1e-5)

                with subtests.test("momentum"):
                    assert norm_momentum == pytest.approx(1e-1)

                with subtests.test("affine"):
                    assert norm_affine

                with subtests.test("track_running_stats"):
                    assert norm_track_running_stats


def test_johnson_alahi_li_2016_conv_block(subtests, mocker):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    for relu in (True, False):
        with subtests.test(relu=relu):
            mock = mocker.patch("torch.nn.Sequential")
            modules.johnson_alahi_li_2016_conv_block(
                in_channels, out_channels, kernel_size, stride=stride, relu=relu
            )

            args, _ = mock.call_args

            assert len(args) == 3 if relu else 2
            assert isinstance(type(args[0]), type(nn.Conv2d))
            assert isinstance(type(args[1]), type(nn.InstanceNorm2d))
            if relu:
                assert isinstance(type(args[2]), type(nn.ReLU))
                assert args[2].inplace


def test_johnson_alahi_li_2016_residual_block(subtests, mocker):
    channels = 3
    mock = mocker.patch("pystiche_papers.utils.ResidualBlock")
    modules.johnson_alahi_li_2016_residual_block(channels)

    for call in mock.call_args_list:
        args, _ = call

        with subtests.test("residual"):
            assert isinstance(type(args[0]), type(nn.Sequential))
            for i in range(0, 2):
                assert isinstance(
                    type(args[0][i]), type(modules.johnson_alahi_li_2016_conv_block)
                )

        with subtests.test("shortcut"):
            assert isinstance(type(args[1]), type(nn.Module))


def test_johnson_alahi_li_2016_transformer_encoder(subtests, mocker):
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
        with subtests.test(instance_norm=instance_norm):
            mock = mocker.patch("pystiche.SequentialModule")
            modules.johnson_alahi_li_2016_transformer_encoder(
                instance_norm=instance_norm
            )

            for call in mock.call_args_list:
                args, _ = call
                in_out_channels = []
                assert isinstance(type(args[0]), type(nn.ReflectionPad2d))
                for i in range(1, 4):
                    assert isinstance(type(args[i]), type(nn.Sequential))
                    in_out_channels.append(
                        (args[i][0].in_channels, args[i][-3].out_channels)
                    )

                for i in range(4, 9):
                    assert isinstance(type(args[i]), type(ResidualBlock))
                    in_out_channels.append(
                        (
                            args[i].residual[0][0].in_channels,
                            args[i].residual[-1][-2].out_channels,
                        )
                    )

            assert in_out_channels == channel_config


def test_johnson_alahi_li_2016_transformer_decoder(subtests, mocker):
    channel_configs = [[(64, 32), (32, 16), (16, 3)], [(128, 64), (64, 32), (32, 3)]]

    for instance_norm, channel_config in zip((True, False), channel_configs):
        with subtests.test(instance_norm=instance_norm):
            mock = mocker.patch("pystiche.SequentialModule")
            modules.johnson_alahi_li_2016_transformer_decoder(
                instance_norm=instance_norm
            )

            for call in mock.call_args_list:
                args, _ = call
                in_out_channels = []
                for i in range(2):
                    assert isinstance(type(args[i]), type(nn.Sequential))
                    in_out_channels.append(
                        (args[i][0].in_channels, args[i][-3].out_channels)
                    )

                assert isinstance(type(args[2]), type(nn.Conv2d))
                in_out_channels.append((args[2].in_channels, args[2].out_channels))

                assert in_out_channels == channel_config

                assert isinstance(type(args[3]), type(nn.Module))
