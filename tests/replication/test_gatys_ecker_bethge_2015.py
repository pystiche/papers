import argparse
import unittest.mock
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_ecker_bethge_2015 as paper
from pystiche import misc, optim

from .asserts import assert_dir_exists

PAPER = "gatys_ecker_bethge_2015"


@pytest.fixture(autouse=True)
def replication_dir_manager(make_replication_dir_manager):
    yield from make_replication_dir_manager(PAPER)


@pytest.fixture(scope="module")
def main(module_loader):
    return module_loader(path.join(PAPER, "main.py"))


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


@pytest.fixture
def nst(mocker):
    return mocker.patch("pystiche_papers.gatys_ecker_bethge_2015.nst")


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


def make_patch_target(name):
    return ".".join(("pystiche_papers", "gatys_ecker_bethge_2015", name))


def attach_method_mock(mock, method, **attrs):
    if "name" not in attrs:
        attrs["name"] = f"{mock.name}.{method}()"

    method_mock = unittest.mock.Mock(**attrs)
    mock.attach_mock(method_mock, method)


@pytest.fixture(scope="module")
def vgg_load_weights_mock(module_mocker):
    return module_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="module", autouse=True)
def multi_layer_encoder(module_mocker, vgg_load_weights_mock):
    multi_layer_encoder = paper.multi_layer_encoder()

    def trim_mock(*args, **kwargs):
        pass

    multi_layer_encoder.trim = trim_mock

    def new(impl_params=None):
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return module_mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015._loss._multi_layer_encoder", new,
    )


@pytest.fixture
def make_images(mocker, image_small_0):
    def make_image_mock(image=None):
        if image is None:
            image = torch.rand_like(image_small_0)
        mock = mocker.Mock()
        attach_method_mock(mock, "read", return_value=image)
        return mock

    def make_images_(*args, **kwargs):
        mocks = {name: make_image_mock() for name in args}
        mocks.update({name: make_image_mock(image) for name, image in kwargs.items()})

        def side_effect(name):
            try:
                return mocks[name]
            except KeyError:
                return unittest.mock.DEFAULT

        images_mock = mocker.Mock()
        attach_method_mock(images_mock, "download")
        attach_method_mock(images_mock, "__getitem__", side_effect=side_effect)
        mocker.patch(make_patch_target("images"), return_value=images_mock)

        return images_mock

    return make_images_


@pytest.fixture
def images(make_images):
    return make_images(*[name for name, _ in paper.images()])


@pytest.fixture(scope="module", autouse=True)
def write_image(module_mocker):
    return module_mocker.patch("pystiche.image.write_image")


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
