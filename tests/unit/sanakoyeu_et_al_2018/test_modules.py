import contextlib
import unittest.mock

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc
from pystiche.enc import Encoder, SequentialEncoder
from pystiche_papers.sanakoyeu_et_al_2018._modules import select_url
from pystiche_papers.utils import AutoPadAvgPool2d, AutoPadConv2d, ResidualBlock

from tests.mocks import make_mock_target
from tests.utils import call_args_to_kwargs_only, generate_param_combinations


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


def test_encoder(subtests):
    channel_config = [(3, 32), (32, 32), (32, 64), (64, 128), (128, 256)]

    for impl_params in (True, False):
        with subtests.test(impl_params):
            encoder = paper.encoder(impl_params=impl_params)
            assert isinstance(encoder, Encoder)

            modules = encoder.children()

            if impl_params:
                assert isinstance(next(modules), nn.InstanceNorm2d)

            assert isinstance(next(modules), nn.ReflectionPad2d)

            in_out_channels = []
            for module in modules:
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
            in_out_channels.append((module.in_channels, module.out_channels))

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


def test_Transformer_smoke(subtests, image_large):
    transformer = paper.Transformer()
    output_image = transformer(image_large)

    with subtests.test("encoder"):
        assert isinstance(transformer.encoder, SequentialEncoder)

    with subtests.test("decoder"):
        assert isinstance(transformer.decoder, pystiche.SequentialModule)

    with subtests.test("forward size"):
        assert image_large.size() == output_image.size()


@pytest.fixture(scope="module")
def model_url_configs(styles):
    return tuple(
        generate_param_combinations(
            framework=("pystiche", "tensorflow"),
            style=styles,
            impl_params=(True, False),
        )
    )


def model_url_should_be_available(style, impl_params, framework):
    if framework == "pystiche":
        return False

    return impl_params


def test_select_url(subtests, model_url_configs):
    for config in model_url_configs:
        with subtests.test(**config):
            if model_url_should_be_available(**config):
                assert isinstance(select_url(**config), str)
            else:
                with pytest.raises(RuntimeError):
                    select_url(**config)


def test_transformer():
    transformer = paper.transformer()
    assert isinstance(transformer, paper.Transformer)


def test_transformer_pretrained(subtests):
    @contextlib.contextmanager
    def patch(target, **kwargs):
        target = make_mock_target("sanakoyeu_et_al_2018", "_modules", target)
        with unittest.mock.patch(target, **kwargs) as mock:
            yield mock

    @contextlib.contextmanager
    def patch_select_url(url):
        with patch("select_url", return_value=url) as mock:
            yield mock

    @contextlib.contextmanager
    def patch_load_state_dict_from_url(state_dict):
        with patch("load_state_dict_from_url", return_value=state_dict) as mock:
            yield mock

    style = "style"
    framework = "framework"
    url = "url"
    for impl_params in (True, False):
        state_dict = paper.Transformer(impl_params=impl_params).state_dict()
        with subtests.test(impl_params=impl_params), patch_select_url(
            url
        ) as select_url, patch_load_state_dict_from_url(state_dict):
            transformer = paper.transformer(
                style=style, impl_params=impl_params, framework=framework
            )

            with subtests.test("select_url"):
                kwargs = call_args_to_kwargs_only(
                    select_url.call_args, "style", "framework", "impl_params",
                )
                assert kwargs["framework"] == framework
                assert kwargs["style"] == style
                assert kwargs["impl_params"] is impl_params

            ptu.assert_allclose(transformer.state_dict(), state_dict)


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
