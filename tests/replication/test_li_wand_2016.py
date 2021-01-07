import argparse
import functools
from os import path

import pytest

import pytorch_testing_utils as ptu

import pystiche_papers.li_wand_2016 as paper
from pystiche import misc, optim

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target, mock_images, patch_multi_layer_encoder_loader
from tests.utils import call_args_list_to_dict

PAPER = "li_wand_2016"


@pytest.fixture(autouse=True)
def dir_manager():
    with utils.dir_manager(PAPER) as dm:
        yield dm


make_paper_mock_target = functools.partial(make_mock_target, PAPER)


@pytest.fixture(scope="module", autouse=True)
def multi_layer_encoder(module_mocker):
    return patch_multi_layer_encoder_loader(
        targets=make_paper_mock_target("_loss", "_multi_layer_encoder"),
        loader=paper.multi_layer_encoder,
        setups=((), {}),
        mocker=module_mocker,
    )


@pytest.fixture
def images(mocker):
    mock = mock_images(mocker, *[name for name, _ in paper.images()])
    mocker.patch(make_paper_mock_target("images"), return_value=mock)
    return mock


@pytest.fixture
def nst(mocker):
    def side_effect(content_image, *args, **kwargs):
        return content_image

    return mocker.patch(make_paper_mock_target("nst"), side_effect=side_effect)


@pytest.fixture(scope="module")
def main():
    return utils.load_module(path.join(PAPER, "main.py"))


@pytest.fixture
def args(tmpdir):
    return argparse.Namespace(
        image_source_dir=tmpdir,
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


def test_figure_6_smoke(subtests, images, nst, main, args):
    main.figure_6(args)

    assert nst.call_count == 2

    call_args_dict = call_args_list_to_dict(
        nst.call_args_list,
        {"top": images["blue_bottle"].read(), "bottom": images["s"].read()},
        args_idx=0,
    )

    with subtests.test("top"):
        call_args = call_args_dict["top"]
        args, _ = call_args
        content_image, style_image = args

        with subtests.test("content_image"):
            ptu.assert_allclose(content_image, images["blue_bottle"].read())

        with subtests.test("style_image"):
            ptu.assert_allclose(style_image, images["self-portrait"].read())

    with subtests.test("bottom"):
        call_args = call_args_dict["bottom"]
        args, _ = call_args
        content_image, style_image = args

        with subtests.test("content_image"):
            ptu.assert_allclose(content_image, images["s"].read())

        with subtests.test("style_image"):
            ptu.assert_allclose(style_image, images["composition_viii"].read())
