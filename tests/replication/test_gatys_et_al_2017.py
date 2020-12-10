import argparse
import functools
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche import misc, optim

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target, mock_images, patch_multi_layer_encoder_loader

PAPER = "gatys_et_al_2017"


@pytest.fixture(autouse=True)
def dir_manager():
    with utils.dir_manager(PAPER) as dm:
        yield dm


make_paper_mock_target = functools.partial(make_mock_target, PAPER)


@pytest.fixture(scope="module", autouse=True)
def multi_layer_encoder(module_mocker):
    return patch_multi_layer_encoder_loader(
        targets=[
            make_paper_mock_target("_loss", "_multi_layer_encoder"),
            make_paper_mock_target("_utils", "multi_layer_encoder_"),
        ],
        loader=paper.multi_layer_encoder,
        setups=((), {}),
        mocker=module_mocker,
    )


@pytest.fixture
def images(mocker):
    mock = mock_images(mocker, *[name for name, _ in paper.images()])
    for name in ("house", "watertown", "wheat_field"):
        image = mock[name].read()
        dark = torch.mean(image, dim=1) < 0.5
        light = ~dark
        mock[name].guides = mock_images(
            mocker, building=dark.float(), sky=light.float()
        )
    mocker.patch(make_paper_mock_target("images"), return_value=mock)
    return mock


@pytest.fixture
def nst(mocker):
    def side_effect(content_image, *args, **kwargs):
        return content_image

    return mocker.patch(make_paper_mock_target("nst"), side_effect=side_effect)


@pytest.fixture
def guided_nst(mocker):
    return mocker.patch(make_paper_mock_target("guided_nst"))


@pytest.fixture(scope="module")
def main():
    return utils.load_module(path.join(PAPER, "main.py"))


@pytest.fixture
def args(tmpdir):
    return argparse.Namespace(
        image_source_dir=tmpdir,
        image_guides_dir=tmpdir,
        image_results_dir=tmpdir,
        device=misc.get_device(),
        impl_params=True,
        logger=optim.OptimLogger(),
        quiet=True,
    )


def test_parse_input_smoke(subtests, main, args):
    actual_args = main.parse_input()

    assert set(vars(actual_args)) == set(vars(args))

    with subtests.test("image_source_dir"):
        assert_dir_exists(actual_args.image_source_dir)

    with subtests.test("image_guides_dir"):
        assert_dir_exists(actual_args.image_guides_dir)

    with subtests.test("image_results_dir"):
        assert_dir_exists(actual_args.image_results_dir)

    with subtests.test("device"):
        assert actual_args.device == args.device

    with subtests.test("impl_params"):
        assert isinstance(actual_args.impl_params, bool)

    with subtests.test("logger"):
        assert isinstance(actual_args.logger, optim.OptimLogger)

    with subtests.test("quiet"):
        assert isinstance(actual_args.quiet, bool)


def test_figure_2_smoke(subtests, images, nst, guided_nst, main, args):
    main.figure_2(args)

    with subtests.test("figure_2_d"):
        assert nst.called_once

        args, _ = nst.call_args
        content_image, style_image = args

        with subtests.test("content_image"):
            ptu.assert_allclose(content_image, images["house"].read())

        with subtests.test("style_image"):
            ptu.assert_allclose(style_image, images["watertown"].read())

    with subtests.test("figure_2_ef"):
        assert guided_nst.call_count == 2

        with subtests.test("content_image"):
            for call_args in guided_nst.call_arg_list:
                args, _ = call_args
                content_image, _, _ = args
                ptu.assert_allclose(content_image, images["house"].read())

        with subtests.test("content_guides"):
            for call_args in guided_nst.call_arg_list:
                args, _ = call_args
                _, content_guides, _ = args
                ptu.assert_allclose(content_guides, images["house"].guides.read())


def test_figure_3_smoke(subtests, images, nst, main, args):
    main.figure_3(args)

    assert nst.call_count == 3
