import argparse
import functools
from os import path

import pytest

from torch.utils.data import TensorDataset

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import misc

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target

PAPER = "sanakoyeu_et_al_2018"


@pytest.fixture(scope="module", autouse=True)
def enable_replication_utils():
    with utils.add_to_sys_path(PAPER):
        yield


@pytest.fixture(autouse=True)
def dir_manager():
    with utils.dir_manager(PAPER) as dm:
        yield dm


make_paper_mock_target = functools.partial(make_mock_target, PAPER)


@pytest.fixture
def content_dataset(mocker):
    return mocker.patch(
        make_paper_mock_target("content_dataset"), return_value=TensorDataset(),
    )


@pytest.fixture
def style_dataset(mocker):
    return mocker.patch(
        make_paper_mock_target("style_dataset"), return_value=TensorDataset()
    )


@pytest.fixture
def training(mocker):
    return mocker.patch(make_paper_mock_target("training"))


@pytest.fixture
def stylization(mocker):
    def side_effect(content_image, *args, **kwargs):
        return content_image

    return mocker.patch(make_paper_mock_target("stylization"), side_effect=side_effect)


@pytest.fixture(scope="module")
def main():
    return utils.load_module(path.join(PAPER, "main.py"))


@pytest.fixture
def args(tmpdir):
    return argparse.Namespace(
        image_source_dir=tmpdir,
        dataset_dir=tmpdir,
        image_results_dir=tmpdir,
        model_dir=tmpdir,
        device=misc.get_device(),
        impl_params=bool,
    )


def test_training_parse_input_smoke(subtests, main, args):
    actual_args = main.parse_input()

    assert set(vars(actual_args)) == set(vars(args))

    with subtests.test("image_source_dir"):
        assert_dir_exists(actual_args.image_source_dir)

    with subtests.test("dataset_dir"):
        assert_dir_exists(actual_args.dataset_dir)

    with subtests.test("image_results_dir"):
        assert_dir_exists(actual_args.image_results_dir)

    with subtests.test("model_dir"):
        assert_dir_exists(actual_args.model_dir)

    with subtests.test("device"):
        assert actual_args.device == args.device

    with subtests.test("impl_params"):
        assert isinstance(actual_args.impl_params, bool)


def test_training_smoke(
    subtests, content_dataset, style_dataset, training, stylization, main, args
):
    main.training(args)

    with subtests.test("content_image_loader"):
        image_loader_type = type(paper.image_loader(content_dataset))
        for call_args in training.call_args_list:
            args, _ = call_args
            content_image_loader, _ = args
            assert isinstance(content_image_loader, image_loader_type)

    with subtests.test("style_image_loader"):
        image_loader_type = type(paper.image_loader(style_dataset))
        for call_args in training.call_args_list:
            args, _ = call_args
            _, style_image_loader = args
            assert isinstance(style_image_loader, image_loader_type)
