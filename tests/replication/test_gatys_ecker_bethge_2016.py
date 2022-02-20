import argparse
import functools
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche import misc, optim
from pystiche_papers.utils import HyperParameters

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
        targets=[
            make_paper_mock_target("_loss", "_multi_layer_encoder"),
            make_paper_mock_target("_utils", "multi_layer_encoder_"),
        ],
        loader=paper.multi_layer_encoder,
        setups=((), {"impl_params": True}),
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


def test_figure_2_smoke(subtests, images, nst, main, args):
    main.figure_2(args)

    assert nst.call_count == 5

    with subtests.test("content_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            content_image, _ = args
            ptu.assert_allclose(content_image, images["neckarfront"].read())

    with subtests.test("style_image"):
        # TODO: make this more precise and also check score weight
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, style_image = args
            assert isinstance(style_image, torch.Tensor)

    with subtests.test("hyper_parameters"):
        for call_args in nst.call_args_list:
            _, kwargs = call_args
            hyper_parameters = kwargs["hyper_parameters"]
            assert isinstance(hyper_parameters, HyperParameters)


def test_figure_3_smoke(subtests, images, nst, main, args):
    main.figure_3(args)

    assert nst.call_count == 20

    with subtests.test("content_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            content_image, _ = args
            ptu.assert_allclose(content_image, images["neckarfront"].read())

    with subtests.test("style_image"):
        for call_args in nst.call_args_list:
            args, _ = call_args
            _, style_image = args
            ptu.assert_allclose(style_image, images["composition_vii"].read())

    with subtests.test("hyper_parameters"):
        for call_args in nst.call_args_list:
            _, kwargs = call_args
            hyper_parameters = kwargs["hyper_parameters"]
            assert isinstance(hyper_parameters, HyperParameters)
