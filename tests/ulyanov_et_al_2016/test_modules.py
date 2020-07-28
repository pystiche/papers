import itertools
from collections import OrderedDict

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

from pystiche.image.utils import extract_num_channels
from pystiche_papers.ulyanov_et_al_2016 import modules


def test_SequentialWithOutChannels(subtests):
    configs = (
        (None, (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))),
        (0, (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))),
        (
            "first_conv",
            (OrderedDict(
                [("first_conv", nn.Conv2d(3, 3, 1)), ("last_conv", nn.Conv2d(3, 5, 1))]
            ),),
        ),
    )
    for out_channel_name, sequential_modules in configs:
        with subtests.test(out_channel_name=out_channel_name):
            module = modules.SequentialWithOutChannels(
                *sequential_modules, out_channel_name=out_channel_name
            )

            output_idx = -1 if out_channel_name is None else out_channel_name
            assert module.out_channels == sequential_modules[output_idx].out_channels


def test_join_channelwise(subtests, image_small_0, image_small_1):
    join_image = modules.join_channelwise(image_small_0, image_small_1)
    assert isinstance(join_image, torch.Tensor)
    input_num_channels = extract_num_channels(image_small_0)
    assert extract_num_channels(
        join_image
    ) == input_num_channels + extract_num_channels(image_small_1)
    with subtests.test("input_image"):
        ptu.assert_allclose(join_image[:, :input_num_channels, :, :], image_small_0)
    with subtests.test("style_image"):
        ptu.assert_allclose(join_image[:, input_num_channels:, :, :], image_small_1)


def test_UlyanovEtAl2016StylizationDownsample(subtests):
    module = modules.UlyanovEtAl2016StylizationDownsample()
    assert isinstance(module, nn.AvgPool2d)
    with subtests.test("kernel_size"):
        assert module.kernel_size == 2
    with subtests.test("stride"):
        assert module.stride == 2
    with subtests.test("padding"):
        assert module.padding == 0


def test_UlyanovEtAl2016TextureDownsample(mocker, input_image):
    mock = mocker.patch(
        "pystiche_papers.ulyanov_et_al_2016.modules.TextureNoiseParams.downsample"
    )
    module = modules.UlyanovEtAl2016TextureDownsample()
    module(input_image)
    mock.assert_called_once()


def test_ulyanov_et_al_2016_downsample():
    module = modules.ulyanov_et_al_2016_downsample()
    assert isinstance(module, modules.UlyanovEtAl2016StylizationDownsample)


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


def test_get_activation_module(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        with subtests.test(impl_params=impl_params, instance_norm=instance_norm):
            norm_module = modules.get_activation_module(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(
                type(norm_module),
                type(nn.ReLU) if impl_params and instance_norm else type(nn.LeakyReLU),
            )

            with subtests.test("inplace"):
                assert norm_module.inplace

            if isinstance(norm_module, nn.LeakyReLU):
                with subtests.test("slope"):
                    assert norm_module.negative_slope == pytest.approx(0.01)
