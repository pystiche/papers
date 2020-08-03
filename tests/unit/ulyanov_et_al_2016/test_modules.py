import itertools
from collections import OrderedDict

import pytest

import pytorch_testing_utils as ptu
import torch
import torch.nn.functional as F
from torch import nn

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import image, misc


def test_SequentialWithOutChannels(subtests):
    sequentialpaper = (nn.Conv2d(3, 3, 1), nn.Conv2d(3, 5, 1))
    sequential_module_dict = OrderedDict(
        ((str(idx), module) for idx, module in enumerate(sequentialpaper))
    )
    for out_channel_name, out_channels, args in (
        (None, 5, sequentialpaper),
        (0, 3, sequentialpaper),
        (1, 5, sequentialpaper),
        ("0", 3, (sequential_module_dict,)),
        ("1", 5, (sequential_module_dict,)),
    ):
        with subtests.test(out_channel_name=out_channel_name):
            sequential = paper.SequentialWithOutChannels(
                *args, out_channel_name=out_channel_name
            )
            assert sequential.out_channels == out_channels


def test_join_channelwise(subtests, image_small_0, image_small_1):
    join_image = paper.join_channelwise(image_small_0, image_small_1)
    assert isinstance(join_image, torch.Tensor)

    input_num_channels = image.extract_num_channels(image_small_0)
    assert image.extract_num_channels(
        join_image
    ) == input_num_channels + image.extract_num_channels(image_small_1)
    ptu.assert_allclose(join_image[:, :input_num_channels, :, :], image_small_0)
    ptu.assert_allclose(join_image[:, input_num_channels:, :, :], image_small_1)


def test_NoiseModule(subtests):
    in_channels = 3
    num_noise_channel = 4
    noise_module = paper.NoiseModule(in_channels, num_noise_channels=num_noise_channel)

    assert isinstance(noise_module, nn.Module)

    with subtests.test("in_channels"):
        assert noise_module.in_channels == in_channels

    with subtests.test("out_channels"):
        assert noise_module.out_channels == in_channels + num_noise_channel


def test_StylizationNoise(input_image):
    in_channels = image.extract_num_channels(input_image)
    num_noise_channel = 3
    module = paper.StylizationNoise(in_channels)
    output_image = module(input_image)
    assert isinstance(output_image, torch.Tensor)
    assert image.extract_num_channels(output_image) == in_channels + num_noise_channel


def test_noise():
    in_channels = 3
    module = paper.noise()
    assert isinstance(module, paper.NoiseModule)
    assert module.in_channels == in_channels


def test_StylizationDownsample(subtests):
    module = paper.StylizationDownsample()
    assert isinstance(module, nn.AvgPool2d)
    with subtests.test("kernel_size"):
        assert module.kernel_size == 2
    with subtests.test("stride"):
        assert module.stride == 2
    with subtests.test("padding"):
        assert module.padding == 0


def test_TextureDownsample(mocker, input_image):
    mock = mocker.patch(
        "pystiche_papers.ulyanov_et_al_2016._modules.TextureNoiseParams.downsample"
    )
    module = paper.TextureDownsample()
    module(input_image)
    mock.assert_called_once()


def test_downsample():
    module = paper.downsample()
    assert isinstance(module, paper.StylizationDownsample)


def test_upsample(subtests):
    module = paper.upsample()
    assert isinstance(module, nn.Upsample)
    with subtests.test("scale_factor"):
        assert module.scale_factor == pytest.approx(2.0)
    with subtests.test("mode"):
        assert module.mode == "nearest"


def test_HourGlassBlock(subtests):
    intermediate = nn.Conv2d(3, 3, 1)
    hour_glass = paper.HourGlassBlock(intermediate)

    assert isinstance(hour_glass, paper.HourGlassBlock)

    with subtests.test("down"):
        assert isinstance(hour_glass.down, paper.StylizationDownsample)
    with subtests.test("intermediate"):
        assert isinstance(hour_glass.intermediate, nn.Conv2d)
    with subtests.test("up"):
        assert isinstance(hour_glass.up, nn.Upsample)


def test_get_norm_module(subtests):
    in_channels = 3
    for instance_norm in (True, False):
        with subtests.test(instance_norm=instance_norm):
            norm_module = paper.norm(in_channels, instance_norm=instance_norm)

            assert isinstance(
                norm_module, nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
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
            norm_module = paper.activation(
                impl_params=impl_params, instance_norm=instance_norm
            )

            assert isinstance(
                norm_module, nn.ReLU if impl_params and instance_norm else nn.LeakyReLU
            )

            with subtests.test("inplace"):
                assert norm_module.inplace

            if isinstance(norm_module, nn.LeakyReLU):
                with subtests.test("slope"):
                    assert norm_module.negative_slope == pytest.approx(0.01)


def test_ConvBlock(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    padding = (1, 1, 1, 1)
    conv_block = paper.ConvBlock(in_channels, out_channels, kernel_size, stride=stride)

    assert isinstance(conv_block, paper.SequentialWithOutChannels)

    with subtests.test("modules"):
        assert len(conv_block) == 4
        assert isinstance(conv_block[0], nn.ReflectionPad2d)
        with subtests.test("conv_module"):
            assert isinstance(conv_block[1], nn.Conv2d)
            assert conv_block[1].stride == misc.to_2d_arg(stride)
            assert conv_block[1].padding == misc.to_2d_arg(0)
        assert isinstance(conv_block[2], nn.InstanceNorm2d)
        assert isinstance(conv_block[3], nn.ReLU)

    with subtests.test("padding"):
        assert conv_block[0].padding == padding


def test_ConvSequence(subtests):
    in_channels = 3
    out_channels = 6
    kernel_size = 3
    conv_sequence = paper.ConvSequence(in_channels, out_channels)

    assert isinstance(conv_sequence, paper.SequentialWithOutChannels)

    with subtests.test("modules"):
        assert len(conv_sequence) == 3
        for i in range(len(conv_sequence)):
            assert isinstance(conv_sequence[i], paper.ConvBlock)
            assert (
                conv_sequence[i][1].in_channels == in_channels
                if i == 0
                else out_channels
            )
            assert conv_sequence[i].out_channels == out_channels
            assert (
                conv_sequence[i][1].kernel_size == misc.to_2d_arg(kernel_size)
                if i < len(conv_sequence) - 1
                else misc.to_2d_arg(1)
            )


def test_JoinBlock(subtests, input_image):
    input_image1 = input_image
    input_image2 = torch.cat((input_image, input_image), 1)
    branch_in_channels = (
        image.extract_num_channels(input_image1),
        image.extract_num_channels(input_image2),
    )
    channel_dim = 1

    for instance_norm, names in itertools.product(
        (True, False), (("block1", "block2"), None)
    ):
        block = paper.JoinBlock(
            branch_in_channels,
            names=names,
            instance_norm=instance_norm,
            channel_dim=channel_dim,
        )

        with subtests.test("norm_modules"):
            assert len(block.norm_modules) == len(branch_in_channels)
            assert any(
                isinstance(
                    norm_module, nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
                )
                for norm_module in block.norm_modules
            )

        with subtests.test("out_channels"):
            assert block.out_channels == sum(branch_in_channels)

        with subtests.test("channel_dim"):
            assert block.channel_dim == channel_dim

        with subtests.test("forward"):
            inputs = (input_image1, input_image2)
            actual = block(*inputs)
            momentum = block.norm_modules[0].momentum
            assert isinstance(actual, torch.Tensor)
            desired_inputs = tuple(
                F.instance_norm(input_image, momentum=momentum)
                if instance_norm
                else F.batch_norm(
                    input_image,
                    torch.zeros(image.extract_num_channels(input_image)),
                    torch.ones(image.extract_num_channels(input_image)),
                    training=True,
                    momentum=momentum,
                )
                for input_image in inputs
            )

            desired = torch.cat(desired_inputs, 1)
            ptu.assert_allclose(actual, desired)


def test_JoinBlock_num_channels_names_mismatch():
    in_channels = 3
    with pytest.raises(RuntimeError):
        paper.JoinBlock((in_channels, in_channels), names=("block1",))


def test_BranchBlock(subtests, input_image):
    deep_branch = nn.Conv2d(3, 3, 1)
    shallow_branch = nn.Conv2d(3, 3, 1)
    block = paper.BranchBlock(deep_branch, shallow_branch)

    with subtests.test("deep"):
        assert block.deep == deep_branch

    with subtests.test("shallow"):
        assert block.shallow == shallow_branch

    with subtests.test("join_block"):
        assert isinstance(block.join, paper.JoinBlock)

    with subtests.test("out_channels"):
        assert (
            block.out_channels == deep_branch.out_channels + shallow_branch.out_channels
        )

    with subtests.test("forward"):
        actual = block(input_image)
        assert isinstance(actual, torch.Tensor)
        momentum = block.join.norm_modules[0].momentum
        inputs = (deep_branch(input_image), shallow_branch(input_image))
        desired_inputs = tuple(
            F.instance_norm(image, momentum=momentum) for image in inputs
        )

        desired = torch.cat(desired_inputs, 1)
        ptu.assert_allclose(actual, desired)


def test_level_conv_sequence(subtests):
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            module = paper.level(None, impl_params=impl_params)
            assert isinstance(module, paper.SequentialWithOutChannels)
            assert len(module) == 3 if impl_params else 2

            with subtests.test("input"):
                first_module = module[0]
                if impl_params:
                    assert first_module[1].in_channels == 3
                else:
                    assert first_module.out_channels == 6

            with subtests.test("noise"):
                if not impl_params:
                    assert isinstance(module[0], paper.StylizationNoise)


def test_level(subtests):
    deep_branch = nn.Conv2d(3, 3, 1)
    module = paper.level(deep_branch)
    assert isinstance(module, paper.SequentialWithOutChannels)
    assert len(module) == 2

    with subtests.test("branch"):
        branch = module[0]
        assert isinstance(branch, paper.BranchBlock)
        assert isinstance(branch.deep, paper.HourGlassBlock)
        assert isinstance(branch.shallow, paper.SequentialWithOutChannels)

    with subtests.test("output_conv_seq"):
        assert isinstance(module[1], paper.SequentialWithOutChannels)


def test_Transformer(subtests, input_image):
    levels = 5

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformer = paper.Transformer(levels, impl_params=impl_params)

            with subtests.test("pyramid"):
                assert isinstance(transformer[0], paper.SequentialWithOutChannels)

            with subtests.test("output_conv"):
                assert isinstance(
                    transformer[1], nn.Conv2d if impl_params else paper.ConvBlock,
                )
                output_conv = transformer[1] if impl_params else transformer[1][0]
                assert output_conv.out_channels == 3
                assert output_conv.kernel_size == misc.to_2d_arg(1)
                assert output_conv.stride == misc.to_2d_arg(1)

            with subtests.test("forward size"):
                output_image = transformer(input_image)
                assert input_image.size() == output_image.size()


def test_transformer():
    transformer = paper.transformer()
    assert isinstance(transformer, paper.Transformer)
