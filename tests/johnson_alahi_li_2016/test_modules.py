import itertools

import pytest

from pystiche_papers.johnson_alahi_li_2016 import modules


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
