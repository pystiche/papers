import argparse
import functools
from os import path

import pytest

from torch.utils.data import TensorDataset

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import misc, optim

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target, mock_images, patch_multi_layer_encoder_loader

PAPER = "ulyanov_et_al_2016"


@pytest.fixture(scope="module", autouse=True)
def enable_replication_utils():
    with utils.add_to_sys_path(PAPER):
        yield


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
def dataset(mocker):
    return mocker.patch(make_paper_mock_target("dataset"), return_value=TensorDataset())


@pytest.fixture
def training(mocker):
    return mocker.patch(make_paper_mock_target("training"))


@pytest.fixture(scope="module")
def main():
    return utils.load_module(path.join(PAPER, "main.py"))


@pytest.fixture
def args(tmpdir):
    return argparse.Namespace(
        image_source_dir=tmpdir,
        image_results_dir=tmpdir,
        dataset_dir=tmpdir,
        model_dir=tmpdir,
        device=misc.get_device(),
        impl_params=bool,
        instance_norm=bool,
        logger=optim.OptimLogger(),
        quiet=True,
    )


def test_training_parse_input_smoke(subtests, main, args):
    actual_args = main.parse_input()

    assert set(vars(actual_args)) == set(vars(args))

    with subtests.test("image_source_dir"):
        assert_dir_exists(actual_args.image_source_dir)

    with subtests.test("image_results_dir"):
        assert_dir_exists(actual_args.image_results_dir)

    with subtests.test("dataset_dir"):
        assert_dir_exists(actual_args.dataset_dir)

    with subtests.test("model_dir"):
        assert_dir_exists(actual_args.model_dir)

    with subtests.test("device"):
        assert actual_args.device == args.device

    with subtests.test("impl_params"):
        assert isinstance(actual_args.impl_params, bool)

    with subtests.test("instance_norm"):
        assert isinstance(actual_args.instance_norm, bool)

    with subtests.test("logger"):
        assert isinstance(actual_args.logger, optim.OptimLogger)

    with subtests.test("quiet"):
        assert isinstance(actual_args.quiet, bool)
