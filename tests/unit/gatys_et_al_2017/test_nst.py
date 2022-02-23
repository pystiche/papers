import unittest.mock

import pytest

import pytorch_testing_utils as ptu
from torchvision.transforms import functional as F

import pystiche_papers.gatys_et_al_2017 as paper

from tests.utils import is_callable


def make_patch_target(name):
    return ".".join(("pystiche_papers", "gatys_et_al_2017", "_nst", name))


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
    def patcher_(name, **kwargs):
        return mocker.patch(make_patch_target(name), **kwargs)

    return patcher_


@pytest.fixture
def preprocessor_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(side_effect=lambda image: image - 0.5)
    patch = patcher("_preprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def postprocessor_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(side_effect=lambda image: image + 0.5)
    patch = patcher("_postprocessor", return_value=mock)
    return patch, mock


@pytest.fixture
def image_pyramid_mocks(mocker, patcher):
    def resize(image_or_guide):
        return F.resize(
            image_or_guide, [length * 2 for length in image_or_guide.shape[-2:]]
        )

    top_level_mock = mocker.Mock()
    attach_method_mock(
        top_level_mock, "resize_image", side_effect=lambda image: resize(image)
    )
    attach_method_mock(
        top_level_mock, "resize_guide", side_effect=lambda guide: resize(guide)
    )
    image_pyramid_mock = mocker.Mock()

    def getitem_side_effect(idx):
        if idx != -1:
            return unittest.mock.DEFAULT

        return top_level_mock

    attach_method_mock(
        image_pyramid_mock, "__getitem__", side_effect=getitem_side_effect
    )
    image_pyramid_patch = patcher("_image_pyramid", return_value=image_pyramid_mock)
    return image_pyramid_patch, image_pyramid_mock, top_level_mock


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
    return patcher("optim.pyramid_image_optimization")


@pytest.mark.slow
def test_nst_smoke(
    subtests, default_image_pyramid_optim_loop_patch, content_image, style_image
):
    patch = default_image_pyramid_optim_loop_patch

    paper.nst(content_image, style_image)

    args, kwargs = patch.call_args
    input_image, criterion, image_pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, image_pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.perceptual_loss()))

    with subtests.test("image_pyramid"):
        assert isinstance(image_pyramid, type(paper.image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(optimizer, type(paper.optimizer(input_image)))

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(paper.preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(paper.postprocessor()))


@pytest.mark.slow
def test_guided_nst_smoke(
    subtests,
    default_image_pyramid_optim_loop_patch,
    content_image,
    content_guides,
    style_images_and_guides,
):
    patch = default_image_pyramid_optim_loop_patch

    paper.guided_nst(content_image, content_guides, style_images_and_guides)

    args, kwargs = patch.call_args
    input_image, criterion, image_pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, image_pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(
            criterion, type(paper.guided_perceptual_loss(content_guides.keys())),
        )

    with subtests.test("image_pyramid"):
        assert isinstance(image_pyramid, type(paper.image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(optimizer, type(paper.optimizer(input_image)))

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(paper.preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(paper.postprocessor()))


def test_guided_nst_regions_mismatch(
    content_image, content_guides, style_images_and_guides
):
    style_images_and_guides.pop(tuple(content_guides.keys())[0])

    with pytest.raises(RuntimeError):
        paper.guided_nst(content_image, content_guides, style_images_and_guides)


def test_guided_nst_device(
    subtests,
    preprocessor_mocks,
    postprocessor_mocks,
    guided_perceptual_loss_mocks,
    content_image,
    content_guides,
    style_images_and_guides,
):
    paper.guided_nst(content_image, content_guides, style_images_and_guides)

    for mocks in (
        preprocessor_mocks,
        postprocessor_mocks,
        guided_perceptual_loss_mocks,
    ):
        _, mock = mocks
        mock = mock.to
        with subtests.test(mock.name):
            mock.assert_called_once_with(content_image.device)


def test_guided_nst_criterion_images_and_guides(
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

    paper.guided_nst(content_image, content_guides, style_images_and_guides)

    patch.assert_called_once()

    args, _ = patch.call_args
    criterion = args[1]

    with subtests.test("content_image"):
        ptu.assert_allclose(
            criterion.content_loss.target_image,
            preprocessor(top_level.resize_image(content_image)),
        )

    with subtests.test("content_guides"):
        for region, content_guide in content_guides.items():
            ptu.assert_allclose(
                criterion.regional_content_guide(region),
                top_level.resize_guide(content_guide),
            )

    with subtests.test("style_images_and_guides"):
        for region, (style_image, style_guide) in style_images_and_guides.items():
            ptu.assert_allclose(
                criterion.regional_style_image(region),
                preprocessor(top_level.resize_image(style_image)),
            )
            ptu.assert_allclose(
                criterion.regional_style_guide(region),
                top_level.resize_guide(style_guide),
            )
