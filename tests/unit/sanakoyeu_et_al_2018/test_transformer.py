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
from pystiche_papers.sanakoyeu_et_al_2018._transformer import select_url
from pystiche_papers.utils import AutoPadConv2d, ResidualBlock

from tests.mocks import make_mock_target
from tests.utils import call_args_to_kwargs_only, generate_param_combinations


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
        target = make_mock_target("sanakoyeu_et_al_2018", "_transformer", target)
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
