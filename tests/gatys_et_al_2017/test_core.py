import unittest.mock

import pytest

import pytorch_testing_utils as ptu

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche.image.transforms.functional import rescale
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
    mock = make_nn_module_mock(side_effect=lambda image: image - 0.5)
    patch = patcher("preprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def postprocessor_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(side_effect=lambda image: image + 0.5)
    patch = patcher("postprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def image_pyramid_mocks(mocker, patcher):
    def resize(image_or_guide):
        return rescale(image_or_guide, 2.0)

    top_level_mock = mocker.Mock()
    attach_method_mock(
        top_level_mock, "resize_image", side_effect=lambda image: resize(image)
    )
    attach_method_mock(
        top_level_mock, "resize_guide", side_effect=lambda guide: resize(guide)
    )
    pyramid_mock = mocker.Mock()

    def getitem_side_effect(idx):
        if idx != -1:
            return unittest.mock.DEFAULT

        return top_level_mock

    attach_method_mock(pyramid_mock, "__getitem__", side_effect=getitem_side_effect)
    pyramid_patch = patcher("image_pyramid", return_value=pyramid_mock)
    return pyramid_patch, pyramid_mock, top_level_mock


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


@pytest.mark.slow
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


@pytest.mark.slow
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
    preprocessor_mocks,
    postprocessor_mocks,
    guided_perceptual_loss_mocks,
    content_image,
    content_guides,
    style_images_and_guides,
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


def test_gatys_et_al_2017_guided_nst_criterion_images_and_guides(
    subtests,
    preprocessor_mocks,
    postprocessor_mocks,
    image_pyramid_mocks,
    default_image_pyramid_optim_loop_patch,
    content_image,
    content_guides,
    style_images_and_guides,
):
    _, _, top_level = image_pyramid_mocks
    _, preprocessor = preprocessor_mocks
    patch = default_image_pyramid_optim_loop_patch

    paper.gatys_et_al_2017_guided_nst(
        content_image, content_guides, style_images_and_guides
    )

    patch.assert_called_once()

    args, _ = patch.call_args
    criterion = args[1]

    with subtests.test("content_image"):
        ptu.assert_allclose(
            criterion.content_loss.target_image,
            preprocessor(top_level.resize_image(content_image)),
        )

    with subtests.test("content_guides"):
        for region, op in criterion.style_loss.named_operators():
            content_guide = content_guides[region]
            ptu.assert_allclose(
                op.get_input_guide(), top_level.resize_guide(content_guide)
            )

    with subtests.test("style_images"):
        for region, op in criterion.style_loss.named_operators():
            style_image, _ = style_images_and_guides[region]
            ptu.assert_allclose(
                op.get_target_image(), preprocessor(top_level.resize_image(style_image))
            )

    with subtests.test("style_guides"):
        for region, op in criterion.style_loss.named_operators():
            _, style_guide = style_images_and_guides[region]
            ptu.assert_allclose(
                op.get_target_guide(), top_level.resize_guide(style_guide)
            )
