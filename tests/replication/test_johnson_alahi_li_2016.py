import argparse
import functools
from collections.abc import Iterable
from os import path

import pytest

import torch
from torch.utils.data import TensorDataset

import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import misc

from . import utils
from .asserts import assert_dir_exists
from tests.mocks import make_mock_target, mock_images, patch_multi_layer_encoder_loader

PAPER = "johnson_alahi_li_2016"


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
        setups=((), {"impl_params": True}),
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


@pytest.fixture
def stylization(mocker):
    return mocker.patch(make_paper_mock_target("stylization"))


@pytest.fixture(scope="module")
def training_script():
    return utils.load_module(path.join(PAPER, "training.py"))


@pytest.fixture
def training_args(tmpdir, patch_argv):
    patch_argv("training.py")
    return argparse.Namespace(
        style=[
            "starry_night",
            "la_muse",
            "composition_vii",
            "the_wave",
            "candy",
            "udnie",
            "the_scream",
            "mosaic",
            "feathers",
        ],
        images_source_dir=tmpdir,
        models_dir=tmpdir,
        dataset_dir=tmpdir,
        impl_params=bool,
        instance_norm=bool,
        device=misc.get_device(),
    )


@pytest.fixture(scope="module")
def stylization_script():
    return utils.load_module(path.join(PAPER, "stylization.py"))


@pytest.fixture
def stylization_args(tmpdir, patch_argv):
    style = "candy"
    patch_argv("stylization.py", style)
    return argparse.Namespace(
        style=style,
        content=["chicago", "hoovertowernight"],
        images_source_dir=tmpdir,
        images_results_dir=tmpdir,
        models_dir=tmpdir,
        impl_params=True,
        instance_norm=True,
        device=misc.get_device(),
    )


def test_training_parse_args_smoke(subtests, training_script, training_args):
    actual_args = training_script.parse_args()

    assert set(vars(actual_args)) == set(vars(training_args))

    with subtests.test("style"):
        assert isinstance(actual_args.style, Iterable)
        assert all(isinstance(style, str) for style in actual_args.style)

    with subtests.test("images_source_dir"):
        assert_dir_exists(actual_args.images_source_dir)

    with subtests.test("models_dir"):
        assert_dir_exists(actual_args.models_dir)

    with subtests.test("dataset_dir"):
        assert_dir_exists(actual_args.dataset_dir)

    with subtests.test("impl_params"):
        assert isinstance(actual_args.impl_params, bool)

    with subtests.test("instance_norm"):
        assert isinstance(actual_args.instance_norm, bool)

    with subtests.test("device"):
        assert actual_args.device == training_args.device


def test_training_main_smoke(
    subtests, images, dataset, training, training_script, training_args
):
    training_script.main(training_args)

    assert training.call_count == len(training_args.style)

    with subtests.test("content_image_loader"):
        image_loader_type = type(paper.image_loader(dataset))
        for call_args in training.call_args_list:
            args, _ = call_args
            content_image_loader, _ = args
            assert isinstance(content_image_loader, image_loader_type)

    with subtests.test("style_image"):
        for call_args in training.call_args_list:
            args, _ = call_args
            _, style_image = args
            assert isinstance(style_image, torch.Tensor)


def test_stylization_parse_args_smoke(subtests, stylization_script, stylization_args):
    actual_args = stylization_script.parse_args()

    assert set(vars(actual_args)) == set(vars(stylization_args))

    with subtests.test("style"):
        assert isinstance(stylization_args.style, str)

    with subtests.test("content"):
        assert isinstance(actual_args.style, Iterable)
        assert all(isinstance(content, str) for content in actual_args.content)

    with subtests.test("images_source_dir"):
        assert_dir_exists(actual_args.images_source_dir)

    with subtests.test("images_results_dir"):
        assert_dir_exists(actual_args.images_results_dir)

    with subtests.test("models_dir"):
        assert_dir_exists(actual_args.models_dir)

    with subtests.test("impl_params"):
        assert isinstance(actual_args.impl_params, bool)

    with subtests.test("instance_norm"):
        assert isinstance(actual_args.instance_norm, bool)

    with subtests.test("device"):
        assert actual_args.device == stylization_args.device


def test_stylization_main_smoke(
    subtests, images, dataset, stylization, stylization_script, stylization_args
):
    stylization_script.main(stylization_args)

    assert stylization.call_count == len(stylization_args.content)

    with subtests.test("input_image"):
        for call_args in stylization.call_args_list:
            args, _ = call_args
            input_image, _ = args
            assert isinstance(input_image, torch.Tensor)

    with subtests.test("content_image_loader"):
        transformer_type = type(paper.transformer())
        for call_args in stylization.call_args_list:
            args, _ = call_args
            _, transformer = args
            assert isinstance(transformer, transformer_type)
