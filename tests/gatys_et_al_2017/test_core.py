import unittest.mock

import pytest

import pytorch_testing_utils as ptu

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche_papers.gatys_et_al_2017 import utils
from tests._utils import is_callable


def make_patch_target(name, prefix=True):
    return ".".join(
        (
            "pystiche_papers",
            "gatys_et_al_2017",
            "core",
            ("gatys_et_al_2017_" if prefix else "") + name,
        )
    )


def attach_method_mock(mock, method, **attrs):
    if "name" not in attrs:
        attrs["name"] = f"{mock.name}.{method}()"

    method_mock = unittest.mock.Mock(**attrs)
    mock.attach_mock(method_mock, method)


@pytest.fixture
def make_nn_module_mock(mocker):
    def make_nn_module_mock_(name=None, identity=False, **kwargs):
        attrs = {}
        if name is not None:
            attrs["name"] = name
        if identity:
            attrs["side_effect"] = lambda x: x
        attrs.update(kwargs)

        mock = mocker.Mock(**attrs)

        for method in ("eval", "to", "train"):
            attach_method_mock(mock, method, return_value=mock)

        return mock

    return make_nn_module_mock_


@pytest.fixture
def patcher(mocker):
    def patcher_(name, prefix=True, **kwargs):
        return mocker.patch(make_patch_target(name, prefix=prefix), **kwargs)

    return patcher_


@pytest.fixture
def preprocessor_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("preprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def postprocessor_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("postprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def guided_perceptual_loss_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock()
    attach_method_mock(mock, "set_content_image", return_value=None)
    attach_method_mock(mock, "set_content_guide", return_value=None)
    attach_method_mock(mock, "set_style_image", return_value=None)
    attach_method_mock(mock, "set_style_guide", return_value=None)
    patch = patcher("guided_perceptual_loss", return_value=mock)
    return patch, mock


@pytest.fixture(autouse=True)
def default_image_pyramid_optim_loop_patch(patcher):
    return patcher("default_image_pyramid_optim_loop", prefix=False)


def test_gatys_et_al_2017_nst_smoke(
    subtests, default_image_pyramid_optim_loop_patch, content_image, style_image
):
    patch = default_image_pyramid_optim_loop_patch

    paper.gatys_et_al_2017_nst(content_image, style_image)

    args, kwargs = patch.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.gatys_et_al_2017_perceptual_loss()))

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.gatys_et_al_2017_image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(
            optimizer, type(utils.gatys_et_al_2017_optimizer(input_image))
        )

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(utils.gatys_et_al_2017_preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(utils.gatys_et_al_2017_postprocessor()))


def test_gatys_et_al_2017_guided_nst_smoke(
    subtests,
    default_image_pyramid_optim_loop_patch,
    content_image,
    content_guides,
    style_images_and_guides,
):
    patch = default_image_pyramid_optim_loop_patch

    paper.gatys_et_al_2017_guided_nst(
        content_image, content_guides, style_images_and_guides
    )

    args, kwargs = patch.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(
            criterion,
            type(paper.gatys_et_al_2017_guided_perceptual_loss(content_guides.keys())),
        )

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.gatys_et_al_2017_image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(
            optimizer, type(utils.gatys_et_al_2017_optimizer(input_image))
        )

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(utils.gatys_et_al_2017_preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(utils.gatys_et_al_2017_postprocessor()))


def test_gatys_et_al_2017_guided_nst_regions_mismatch(
    content_image, content_guides, style_images_and_guides
):
    style_images_and_guides.pop(tuple(content_guides.keys())[0])

    with pytest.raises(RuntimeError):
        paper.gatys_et_al_2017_guided_nst(
            content_image, content_guides, style_images_and_guides
        )


def test_gatys_et_al_2017_guided_nst_device(
    subtests,
    content_image,
    content_guides,
    style_images_and_guides,
    preprocessor_mocks,
    postprocessor_mocks,
    guided_perceptual_loss_mocks,
):
    paper.gatys_et_al_2017_guided_nst(
        content_image, content_guides, style_images_and_guides
    )

    for mocks in (
        preprocessor_mocks,
        postprocessor_mocks,
        guided_perceptual_loss_mocks,
    ):
        _, mock = mocks
        mock = mock.to
        with subtests.test(mock.name):
            mock.assert_called_once_with(content_image.device)
