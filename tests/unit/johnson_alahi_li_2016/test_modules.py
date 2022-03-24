import contextlib
import unittest.mock

import pytest

from tests.mocks import make_mock_target
from tests.utils import call_args_to_kwargs_only, generate_param_combinations

import pytorch_testing_utils as ptu
import torch
from torch import nn

import pystiche
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import misc
from pystiche_papers import utils
from pystiche_papers.johnson_alahi_li_2016._modules import select_url


@pytest.fixture(scope="module")
def model_url_configs(styles):
    return tuple(
        generate_param_combinations(
            framework=("pystiche", "luatorch"),
            style=styles,
            impl_params=(True, False),
            instance_norm=(True, False),
        )
    )


def model_url_should_be_available(framework, style, impl_params, instance_norm):
    if framework == "pystiche":
        return True

    if not impl_params:
        return False

    if style in ("composition_vii", "starry_night", "the_wave") and instance_norm:
        return False

    if (
        style in ("candy", "feathers", "mosaic", "the_scream", "udnie")
        and not instance_norm
    ):
        return False

    return True


def test_select_url(subtests, model_url_configs):
    for config in model_url_configs:
        with subtests.test(**config):
            if model_url_should_be_available(**config):
                assert isinstance(select_url(**config), str)
            else:
                with pytest.raises(RuntimeError):
                    select_url(**config)


def test_get_conv(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    for params in generate_param_combinations(
        padding=(None, 1), upsample=(True, False)
    ):
        with subtests.test(**params):
            conv = paper.conv(
                in_channels, out_channels, kernel_size, stride=stride, **params
            )

            assert isinstance(
                conv, nn.ConvTranspose2d if params["upsample"] else nn.Conv2d
            )

            with subtests.test("in_channels"):
                assert conv.in_channels == in_channels

            with subtests.test("out_channels"):
                assert conv.out_channels == out_channels

            with subtests.test("kernel_size"):
                assert conv.kernel_size == misc.to_2d_arg(kernel_size)

            with subtests.test("stride"):
                assert conv.stride == misc.to_2d_arg(stride)

            if params["padding"] is not None:
                with subtests.test("padding"):
                    assert conv.padding == misc.to_2d_arg(params["padding"])


def test_get_norm(subtests):
    out_channels = 3
    for instance_norm in (True, False):
        with subtests.test(instance_norm=instance_norm):
            norm = paper.norm(out_channels, instance_norm=instance_norm)

            assert isinstance(
                norm, nn.InstanceNorm2d if instance_norm else nn.BatchNorm2d
            )

            with subtests.test("out_channels"):
                assert norm.num_features == out_channels

            with subtests.test("eps"):
                assert norm.eps == pytest.approx(1e-5)

            with subtests.test("momentum"):
                assert norm.momentum == pytest.approx(1e-1)

            with subtests.test("affine"):
                assert norm.affine

            with subtests.test("track_running_stats"):
                assert norm.track_running_stats


def test_conv_block(subtests):
    in_channels = out_channels = 3
    kernel_size = 3
    stride = 1
    for params in generate_param_combinations(
        padding=(None, 1),
        upsample=(True, False),
        relu=(True, False),
        inplace=(True, False),
        instance_norm=(True, False),
    ):
        conv_block = paper.conv_block(
            in_channels, out_channels, kernel_size, stride=stride, **params
        )
        assert isinstance(conv_block, nn.Sequential)
        assert len(conv_block) == 3 if params["relu"] else 2

        with subtests.test("conv"):
            assert isinstance(
                conv_block[0],
                type(
                    paper.conv(
                        1, 1, 1, padding=params["padding"], upsample=params["upsample"]
                    )
                ),
            )

        with subtests.test("norm"):
            assert isinstance(
                conv_block[1], type(paper.norm(1, params["instance_norm"]))
            )

        if params["relu"]:
            with subtests.test("relu"):
                assert isinstance(conv_block[2], nn.ReLU)
                assert conv_block[2].inplace is params["inplace"]


def test_residual_block(subtests, input_image):
    channels = 3
    residual_block = paper.residual_block(channels)

    assert isinstance(residual_block, utils.ResidualBlock)

    with subtests.test("residual"):
        assert isinstance(residual_block.residual, nn.Sequential)
        assert len(residual_block.residual) == 2

    with subtests.test("shortcut"):
        assert isinstance(residual_block.shortcut, nn.Module)
        ptu.assert_allclose(
            input_image[:, :, 2:-2, 2:-2], residual_block.shortcut(input_image)
        )


def test_transformer_encoder(subtests):
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

        encoder = paper.encoder(instance_norm=instance_norm)

        assert isinstance(encoder, pystiche.SequentialModule)

        in_out_channels = []
        for i, module in enumerate(encoder.children()):
            if i == 0:
                with subtests.test("padding_module"):
                    assert isinstance(module, nn.ReflectionPad2d)

            if i in range(1, 4):
                with subtests.test("conv_layer"):
                    assert isinstance(module, nn.Sequential)
                    in_out_channels.append(
                        (module[0].in_channels, module[0].out_channels)
                    )
            if i in range(4, 9):
                with subtests.test("residual_blocks"):
                    assert isinstance(module, utils.ResidualBlock)
                    in_out_channels.append(
                        (
                            module.residual[0][0].in_channels,
                            module.residual[-1][0].out_channels,
                        )
                    )

        with subtests.test("channel_config"):
            assert in_out_channels == channel_config


def test_transformer_decoder(subtests):
    channel_configs = [[(64, 32), (32, 16), (16, 3)], [(128, 64), (64, 32), (32, 3)]]

    for instance_norm, channel_config in zip((True, False), channel_configs):
        with subtests.test(instance_norm=instance_norm):
            decoder = paper.decoder(instance_norm=instance_norm)

            assert isinstance(decoder, pystiche.SequentialModule)

            in_out_channels = []
            for i, module in enumerate(decoder.children()):
                if i in range(2):
                    with subtests.test("conv_layer"):
                        assert isinstance(module, nn.Sequential)
                        in_out_channels.append(
                            (module[0].in_channels, module[0].out_channels)
                        )
                if i == 2:
                    with subtests.test("output_conv"):
                        assert isinstance(module, nn.Conv2d)
                        in_out_channels.append(
                            (module.in_channels, module.out_channels)
                        )

            with subtests.test("channel_config"):
                assert in_out_channels == channel_config


def test_transformer_decoder_value_range_delimiter(subtests, input_image):
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            decoder = paper.decoder(impl_params=impl_params)

            module = [x for x in decoder.children()][-1]
            assert isinstance(module, nn.Module)

            with subtests.test("delimiter"):
                actual = module(input_image)
                desired = (
                    150.0 * torch.tanh(input_image)
                    if impl_params
                    else torch.sigmoid(2.0 * input_image)
                )

                ptu.assert_allclose(actual, desired)


def test_Transformer_smoke():
    transformer = paper.Transformer()
    assert isinstance(transformer.encoder, pystiche.SequentialModule)
    assert isinstance(transformer.decoder, pystiche.SequentialModule)


def test_transformer_smoke(subtests, image_medium):
    transformer = paper.transformer()
    assert isinstance(transformer, paper.Transformer)

    with subtests.test("forward size"):
        output_image = transformer(image_medium)
        assert image_medium.size() == output_image.size()


def test_transformer_pretrained(subtests):
    @contextlib.contextmanager
    def patch(target, **kwargs):
        target = make_mock_target("johnson_alahi_li_2016", "_modules", target)
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

    framework = "framework"
    style = "style"
    url = "url"
    for config in generate_param_combinations(
        impl_params=(True, False), instance_norm=(True, False)
    ):
        state_dict = paper.Transformer(**config).state_dict()
        with subtests.test(**config), patch_select_url(
            url
        ) as select_url, patch_load_state_dict_from_url(state_dict):
            transformer = paper.transformer(framework=framework, style=style, **config)

            with subtests.test("select_url"):
                kwargs = call_args_to_kwargs_only(
                    select_url.call_args,
                    "framework",
                    "style",
                    "impl_params",
                    "instance_norm",
                )
                assert kwargs["framework"] == framework
                assert kwargs["style"] == style
                assert kwargs["impl_params"] is config["impl_params"]
                assert kwargs["instance_norm"] is config["instance_norm"]

            ptu.assert_allclose(transformer.state_dict(), state_dict)
