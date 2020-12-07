import itertools
import unittest.mock

import pytest

import pytorch_testing_utils as ptu
import torch
from torch.utils.data import DataLoader, TensorDataset

import pystiche.image.transforms.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche_papers.utils import batch_up_image

from tests import utils
from tests.utils import is_callable


def make_patch_target(name):
    return ".".join(("pystiche_papers", "johnson_alahi_li_2016", "_nst", name))


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
def optimizer_mocks(mocker, patcher):
    mock = mocker.Mock()
    patch = patcher("_optimizer", return_value=mock)
    return patch, mock


@pytest.fixture
def images_patch(mocker, content_image, style_image):
    def make_image_mock(image):
        mock = mocker.Mock()
        attach_method_mock(mock, "read", return_value=image)
        return mock

    content_image_mock = make_image_mock(content_image)
    style_image_mock = make_image_mock(style_image)

    def side_effect(name):
        if name == "content":
            return content_image_mock
        elif name == "style":
            return style_image_mock
        else:
            return unittest.mock.DEFAULT

    images_mock = mocker.Mock()
    attach_method_mock(images_mock, "__getitem__", side_effect=side_effect)
    images_patch = mocker.patch(make_patch_target("_images"), return_value=images_mock)
    return images_patch, images_mock


@pytest.fixture
def style_transforms_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(side_effect=lambda image: F.rescale(image, 2.0))
    patch = patcher("_style_transform", return_value=mock)
    return patch, mock


@pytest.fixture
def transformer_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("_transformer", return_value=mock)
    return patch, mock


@pytest.fixture
def perceptual_loss_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock()
    attach_method_mock(mock, "set_content_image", return_value=None)
    attach_method_mock(mock, "set_style_image", return_value=None)
    patch = patcher("perceptual_loss", return_value=mock)
    return patch, mock


@pytest.fixture
def default_transformer_optim_loop_patch(patcher):
    def side_effect(_, transformer, *args, **kwargs):
        return transformer

    return patcher(
        "optim.default_transformer_optim_loop", prefix=False, side_effect=side_effect
    )


@pytest.fixture
def image_loader(image):
    return DataLoader(TensorDataset(image), batch_size=2)


def reset_mocks(*mocks):
    for mock in mocks:
        mock.reset_mock()


@pytest.fixture
def training(default_transformer_optim_loop_patch, image_loader, style_image):
    def training_(image_loader_=None, style_image_=None, **kwargs):
        if image_loader_ is None:
            image_loader_ = image_loader
        if style_image_ is None:
            style_image_ = style_image
        output = paper.training(image_loader_, style_image_, **kwargs)

        default_transformer_optim_loop_patch.assert_called_once()
        args, kwargs = default_transformer_optim_loop_patch.call_args

        return args, kwargs, output

    return training_


@pytest.mark.slow
def test_training_smoke(subtests, training, image_loader):
    args, kwargs, output = training(image_loader)
    content_image_loader, transformer, criterion, criterion_update_fn = args
    optimizer = kwargs["optimizer"]

    with subtests.test("content_image_loader"):
        assert content_image_loader is image_loader

    with subtests.test("transformer"):
        assert isinstance(transformer, type(paper.transformer()))

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.perceptual_loss()))

    with subtests.test("criterion_update_fn"):
        assert is_callable(criterion_update_fn)

    with subtests.test("optimizer"):
        assert isinstance(optimizer, type(paper.optimizer(transformer)))

    with subtests.test("output"):
        assert output is transformer


@utils.parametrize_data(
    ("impl_params", "instance_norm"),
    *[
        pytest.param(impl_params, instance_norm)
        for impl_params, instance_norm in itertools.product(
            (True, False), (True, False, None)
        )
    ],
)
def test_training_instance_norm(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    default_transformer_optim_loop_patch,
    training,
    impl_params,
    instance_norm,
):
    mock, _ = transformer_mocks
    training(impl_params=impl_params, instance_norm=instance_norm)

    mock.assert_called_once()

    _, kwargs = mock.call_args
    if instance_norm is None:
        assert kwargs["instance_norm"] is impl_params
    else:
        assert kwargs["instance_norm"] is instance_norm


def test_training_device(
    subtests,
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    training,
    style_image,
):
    training(style_image_=style_image)

    for mocks in (
        preprocessor_mocks,
        transformer_mocks,
        perceptual_loss_mocks,
        style_transforms_mocks,
    ):
        _, mock = mocks
        mock = mock.to
        with subtests.test(mock.name):
            mock.assert_called_once_with(style_image.device)


def test_training_transformer_train(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    training,
):
    args, _, _ = training()
    transformer = args[1]

    transformer.train.assert_called_once_with()


def test_training_criterion_eval(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    training,
):
    args, _, _ = training()
    criterion = args[2]

    criterion.eval.assert_called_once_with()


def test_training_criterion_content_image(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    training,
):
    args, _, _ = training()
    criterion = args[2]

    assert not criterion.content_loss.has_target_image


def test_training_criterion_style_image(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    training,
    image_loader,
    style_image,
):
    _, preprocessor = preprocessor_mocks
    _, style_transform = style_transforms_mocks

    args, _, _ = training(image_loader, style_image)
    criterion = args[2]

    ptu.assert_allclose(
        criterion.style_loss.get_target_image(),
        preprocessor(
            style_transform(batch_up_image(style_image, image_loader.batch_size))
        ),
    )


def test_training_criterion_update_fn(
    preprocessor_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    transformer_mocks,
    training,
    content_image,
):
    args, _, _ = training()
    _, _, criterion, criterion_update_fn = args

    assert not criterion.content_loss.has_target_image

    criterion_update_fn(content_image, criterion)
    assert criterion.content_loss.has_target_image
    ptu.assert_allclose(criterion.content_loss.target_image, content_image)


def test_training_optimizer(
    preprocessor_mocks, style_transforms_mocks, transformer_mocks, training,
):
    parameter = torch.empty(1)
    _, transformer_mock = transformer_mocks
    attach_method_mock(transformer_mock, "parameters", return_value=iter((parameter,)))

    _, kwargs, _ = training()
    optimizer = kwargs["optimizer"]

    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 1

    param = param_group["params"][0]
    assert param is parameter


@pytest.fixture
def stylization(input_image, transformer_mocks):
    _, transformer = transformer_mocks

    def stylization_(input_image_=None, transformer_="style", **kwargs):
        if input_image_ is None:
            input_image_ = input_image

        output = paper.stylization(input_image_, transformer_, **kwargs)

        if isinstance(transformer_, str):
            transformer.assert_called_once()
            args, kwargs = transformer.call_args
        else:
            try:
                transformer_.assert_called_once()
                args, kwargs = transformer.call_args
            except AttributeError:
                args = kwargs = None

        return args, kwargs, output

    return stylization_


def test_stylization_smoke(stylization, input_image):
    _, _, output_image = stylization(input_image)
    ptu.assert_allclose(output_image, input_image, rtol=1e-6)


def test_stylization_device(
    subtests,
    preprocessor_mocks,
    postprocessor_mocks,
    transformer_mocks,
    stylization,
    input_image,
):
    stylization(input_image)

    for mocks in (
        preprocessor_mocks,
        postprocessor_mocks,
        transformer_mocks,
    ):
        _, mock = mocks
        mock = mock.to
        with subtests.test(mock.name):
            mock.assert_called_once_with(input_image.device)


def test_stylization_instance_norm(
    subtests,
    preprocessor_mocks,
    postprocessor_mocks,
    transformer_mocks,
    stylization,
    input_image,
):
    transformer_patch, _ = transformer_mocks
    for impl_params, instance_norm in itertools.product(
        (True, False), (True, False, None)
    ):
        reset_mocks(transformer_patch)
        stylization(input_image, impl_params=impl_params, instance_norm=instance_norm)

        with subtests.test(
            transformer_patch.name, impl_params=impl_params, instance_norm=instance_norm
        ):
            transformer_patch.assert_called_once()

            _, kwargs = transformer_patch.call_args
            if instance_norm is None:
                assert kwargs["instance_norm"] is impl_params
            else:
                assert kwargs["instance_norm"] is instance_norm


def test_stylization_transformer_eval(
    subtests, preprocessor_mocks, postprocessor_mocks, transformer_mocks, stylization
):
    _, transformer = transformer_mocks
    stylization()

    transformer.eval.assert_called_once_with()


def test_stylization_transformer_str(
    subtests, preprocessor_mocks, postprocessor_mocks, transformer_mocks, stylization,
):
    patch, mock = transformer_mocks

    style = "style"
    framework = "pystiche"
    stylization(transformer_=style, framework=framework)

    patch.assert_called_once()
    _, kwargs = patch.call_args

    with subtests.test("style"):
        assert kwargs["style"] == style

    with subtests.test("framework"):
        assert kwargs["framework"] == framework

    with subtests.test("eval"):
        mock.eval.assert_called_once_with()
