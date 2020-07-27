import itertools
from typing import Dict, cast

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

from pystiche.image.utils import extract_num_channels
from pystiche_papers.ulyanov_et_al_2016 import modules


def test_SequentialWithOutChannels(subtests):
    sequential_modules = [nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1)]
    for out_channel_name in (
        None,
        1,
        "1",
    ):  # TODO: Rename out_channel_name is not implemented
        module = modules.SequentialWithOutChannels(
            *sequential_modules, out_channel_name=out_channel_name
        )

        with subtests.test("out_channel_name"):
            if out_channel_name is None:
                desired_out_channel_name = "1"
            elif isinstance(out_channel_name, int):
                desired_out_channel_name = str(out_channel_name)
            else:
                desired_out_channel_name = out_channel_name

            assert (
                tuple(cast(Dict[str, nn.Module], module._modules).keys())[-1]
                == desired_out_channel_name
            )

        with subtests.test("out_channels"):
            assert module.out_channels == sequential_modules[-1].out_channels


def test_join_channelwise(subtests, input_image, style_image):
    join_image = modules.join_channelwise(input_image, style_image)
    assert isinstance(join_image, torch.Tensor)
    input_num_channels = extract_num_channels(input_image)
    with subtests.test("num_channels"):
        assert extract_num_channels(
            join_image
        ) == input_num_channels + extract_num_channels(style_image)
    with subtests.test("input_image"):
        ptu.assert_allclose(join_image[:, :input_num_channels, :, :], input_image)
    with subtests.test("style_image"):
        ptu.assert_allclose(join_image[:, input_num_channels:, :, :], style_image)


def test_UlyanovEtAl2016StylizationDownsample(subtests):
    module = modules.UlyanovEtAl2016StylizationDownsample()
    assert isinstance(module, nn.AvgPool2d)
    with subtests.test("kernel_size"):
        assert module.kernel_size == pytest.approx(2)
    with subtests.test("stride"):
        assert module.stride == pytest.approx(2)
    with subtests.test("padding"):
        assert module.padding == pytest.approx(0)


def test_UlyanovEtAl2016TextureDownsample(mocker, input_image):
    mock = mocker.patch(
        "pystiche_papers.ulyanov_et_al_2016.modules.TextureNoiseParams.downsample"
    )
    module = modules.UlyanovEtAl2016TextureDownsample()
    module(input_image)
    mock.assert_called_once()


def test_ulyanov_et_al_2016_downsample(subtests):
    for stylization in (True, False):
        with subtests.test(stylization=stylization):
            module = modules.ulyanov_et_al_2016_downsample(stylization=stylization)
            assert isinstance(
                module,
                modules.UlyanovEtAl2016StylizationDownsample
                if stylization
                else modules.UlyanovEtAl2016TextureDownsample,
            )


def test_ulyanov_et_al_2016_upsample(subtests):
    module = modules.ulyanov_et_al_2016_upsample()
    assert isinstance(module, nn.Upsample)
    with subtests.test("scale_factor"):
        assert module.scale_factor == pytest.approx(2.0)
    with subtests.test("mode"):
        assert module.mode == "nearest"


def test_get_norm_module(subtests):
    in_channels = 3
    for instance_norm in (True, False):
        with subtests.test(instance_norm=instance_norm):
            norm_module = modules.get_norm_module(
                in_channels, instance_norm=instance_norm
            )

            assert isinstance(
                type(norm_module),
                type(nn.InstanceNorm2d) if instance_norm else type(nn.BatchNorm2d),
            )

            with subtests.test("out_channels"):
                assert norm_module.num_features == in_channels

            with subtests.test("eps"):
                assert norm_module.eps == pytest.approx(1e-5)

            with subtests.test("momentum"):
                assert norm_module.momentum == pytest.approx(1e-1)

            with subtests.test("affine"):
                assert norm_module.affine

            with subtests.test("track_running_stats"):
                assert norm_module.track_running_stats


def test_get_activation_module(subtests):
    for instance_norm, impl_params in itertools.product((True, False), (True, False)):
        with subtests.test(instance_norm=instance_norm):
            norm_module = modules.get_activation_module(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(
                type(norm_module),
                type(nn.ReLU) if impl_params and instance_norm else type(nn.LeakyReLU),
            )

            with subtests.test("inplace"):
                assert norm_module.inplace

            if not impl_params and instance_norm:
                with subtests.test("slope"):
                    assert norm_module.negative_slope == pytest.approx(0.01)
