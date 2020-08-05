import argparse
import functools
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche import misc, optim

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target, mock_images, patch_multi_layer_encoder_loader

PAPER = "gatys_ecker_bethge_2016"


@pytest.fixture(autouse=True)
def dir_manager():
    with utils.dir_manager(PAPER) as dm:
        yield dm


make_paper_mock_target = functools.partial(make_mock_target, PAPER)


@pytest.fixture(scope="module", autouse=True)
def multi_layer_encoder(module_mocker):
    return patch_multi_layer_encoder_loader(
        target=make_paper_mock_target("_loss", "_multi_layer_encoder"),
        loader=paper.multi_layer_encoder,
        setup=((), {"impl_params": True}),
        mocker=module_mocker,
    )


@pytest.fixture
def images(mocker):
    mock = mock_images(mocker, *[name for name, _ in paper.images()])
    mocker.patch(make_paper_mock_target("images"), return_value=mock)
    return mock


@pytest.fixture
def nst(mocker):
    return mocker.patch(make_paper_mock_target("nst"))


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


@pytest.mark.slow
def test_figure_2_smoke(subtests, images, nst, main, args):
    main.figure_2(args)

    assert nst.call_count == 5

    with subtests.test("content_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            content_image, _, _ = args
            ptu.assert_allclose(content_image, images["neckarfront"].read())

    with subtests.test("style_image"):
        # TODO: make this more precise and also check score weight
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, style_image, _ = args
            assert isinstance(style_image, torch.Tensor)

    with subtests.test("num_steps"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, _, num_steps = args
            assert num_steps == 500

    with subtests.test("perceptual_loss"):
        criterion_type = type(paper.perceptual_loss())
        for call_args in nst.call_args_list:
            _, kwargs = call_args
            criterion = kwargs["criterion"]
            assert isinstance(criterion, criterion_type)


@pytest.mark.slow
def test_figure_3_smoke(subtests, images, nst, main, args):
    main.figure_3(args)

    assert nst.call_count == 20

    with subtests.test("content_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            content_image, _, _ = args
            ptu.assert_allclose(content_image, images["neckarfront"].read())

    with subtests.test("style_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, style_image, _ = args
            ptu.assert_allclose(style_image, images["composition_vii"].read())

    with subtests.test("num_steps"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, _, num_steps = args
            assert num_steps == 500

    with subtests.test("perceptual_loss"):
        criterion_type = type(paper.perceptual_loss())
        for call_args in nst.call_args_list:
            _, kwargs = call_args
            criterion = kwargs["criterion"]
            assert isinstance(criterion, criterion_type)
