import itertools
import unittest.mock

import pytest

import pytorch_testing_utils as ptu
from torch.utils.data import DataLoader, TensorDataset

import pystiche_papers.johnson_alahi_li_2016 as paper

from .._utils import is_callable


def make_patch_target(name, prefix=True):
    return ".".join(
        (
            "pystiche_papers",
            "johnson_alahi_li_2016",
            "core",
            ("johnson_alahi_li_2016_" if prefix else "") + name,
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
def transformer_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock()
    patch = patcher("transformer", return_value=mock)
    return patch, mock


@pytest.fixture
def perceptual_loss_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock()
    attach_method_mock(mock, "set_content_image", return_value=None)
    attach_method_mock(mock, "set_style_image", return_value=None)
    patch = patcher("perceptual_loss", return_value=mock)
    return patch, mock


@pytest.fixture
def optimizer_mocks(mocker, patcher):
    mock = mocker.Mock()
    patch = patcher("optimizer", return_value=mock)
    return patch, mock


@pytest.fixture
def style_transforms_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("style_transform", return_value=mock)
    return patch, mock


@pytest.fixture
def default_transformer_optim_loop_patch(patcher):
    return patcher("default_transformer_optim_loop", prefix=False)


@pytest.fixture
def image_loader(image):
    return DataLoader(TensorDataset(image))


def reset_mocks(*mocks):
    for mock in mocks:
        mock.reset_mock()


def test_johnson_alahi_li_2016_training_content_image_loader(
    image_loader,
    style_image,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
):
    patch = default_transformer_optim_loop_patch

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    patch.assert_called_once()

    args, _ = patch.call_args
    assert args[0] is image_loader


def test_johnson_alahi_li_2016_training_style_image(
    subtests,
    patcher,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    patch = patcher("batch_up_image", prefix=False)

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    for mock in (preprocessor_mocks[1], style_transforms_mocks[1], patch):
        with subtests.test(mock.name):
            patch.assert_called_once()

            args, _ = mock.call_args
            ptu.assert_allclose(args[0], style_image)


def test_johnson_alahi_li_2016_training_style_image_not_impl_params(
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    paper.johnson_alahi_li_2016_training(image_loader, style_image, impl_params=False)

    _, mock = preprocessor_mocks
    mock.assert_not_called()


def test_johnson_alahi_li_2016_training_style_image_tensor(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    with subtests.test("style"):
        for mocks in (perceptual_loss_mocks, style_transforms_mocks):
            patch, _ = mocks
            with subtests.test(patch.name):
                patch.assert_called_once()

                _, kwargs = patch.call_args
                assert kwargs["style"] is None

    with subtests.test("device"):
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


def test_johnson_alahi_li_2016_training_style_image_str(
    subtests,
    mocker,
    patcher,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    image_mock = mocker.Mock()
    image_mock.attach_mock(mocker.Mock(return_value=style_image), "read")
    images_mock = mocker.Mock()
    images_mock.attach_mock(mocker.Mock(return_value=image_mock), "__getitem__")
    patch = mocker.patch(make_patch_target("images"), return_value=images_mock)

    style = "style"
    paper.johnson_alahi_li_2016_training(image_loader, style)

    patch.assert_called_once()

    with subtests.test("style_image"):
        _, mock = style_transforms_mocks
        mock.assert_called_once()

        args, _ = mock.call_args
        ptu.assert_allclose(args[0], style_image)

    with subtests.test("style"):
        for mocks in (perceptual_loss_mocks, style_transforms_mocks):
            patch, _ = mocks
            with subtests.test(patch.name):
                patch.assert_called_once()

                _, kwargs = patch.call_args
                assert kwargs["style"] == style


def test_johnson_alahi_li_2016_training_instance_norm_default(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    mocks = (transformer_mocks[0], perceptual_loss_mocks[0], style_transforms_mocks[0])

    for impl_params in (True, False):
        reset_mocks(*mocks)
        paper.johnson_alahi_li_2016_training(
            image_loader, style_image, impl_params=impl_params, instance_norm=None
        )

        for mock in mocks:
            with subtests.test(mock.name, impl_params=impl_params):
                mock.assert_called_once()

                _, kwargs = mock.call_args
                assert kwargs["instance_norm"] is impl_params


def test_johnson_alahi_li_2016_training_instance_norm(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    mocks = (transformer_mocks[0], perceptual_loss_mocks[0], style_transforms_mocks[0])

    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        reset_mocks(*mocks)
        paper.johnson_alahi_li_2016_training(
            image_loader,
            style_image,
            impl_params=impl_params,
            instance_norm=instance_norm,
        )

        for mock in mocks:
            with subtests.test(
                mock.name, impl_params=impl_params, instance_norm=instance_norm
            ):
                mock.assert_called_once()

                _, kwargs = mock.call_args
                assert kwargs["instance_norm"] is instance_norm


def test_johnson_alahi_li_2016_training_transformer(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    _, mock = transformer_mocks

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    default_transformer_optim_loop_patch.assert_called_once()
    args, _ = default_transformer_optim_loop_patch.call_args

    assert args[1] is mock

    with subtests.test("train"):
        mock.train.assert_called_once_with()

    with subtests.test("to"):
        mock.to.assert_called_once_with(style_image.device)


def test_johnson_alahi_li_2016_training_criterion(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    _, mock = perceptual_loss_mocks

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    default_transformer_optim_loop_patch.assert_called_once()
    args, _ = default_transformer_optim_loop_patch.call_args

    assert args[2] is mock

    with subtests.test("eval"):
        mock.eval.assert_called_once_with()

    with subtests.test("to"):
        mock.to.assert_called_once_with(style_image.device)

    with subtests.test("set_style_image"):
        mock.set_style_image.assert_called_once()

        args, _ = mock.set_style_image.call_args
        ptu.assert_allclose(args[0], style_image)


def test_johnson_alahi_li_2016_training_criterion_update_fn(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    content_image,
    style_image,
):
    _, mock = perceptual_loss_mocks
    patch = default_transformer_optim_loop_patch

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    patch.assert_called_once()

    args, _ = patch.call_args
    criterion_update_fn = args[3]

    assert is_callable(criterion_update_fn)

    with subtests.test("update"):
        mock.set_content_image.assert_not_called()

        criterion_update_fn(content_image, mock)

        mock.set_content_image.assert_called_once()

        args, _ = mock.set_content_image.call_args
        ptu.assert_allclose(args[0], content_image)


def test_johnson_alahi_li_2016_training_optimizer(
    subtests,
    preprocessor_mocks,
    transformer_mocks,
    perceptual_loss_mocks,
    optimizer_mocks,
    style_transforms_mocks,
    default_transformer_optim_loop_patch,
    image_loader,
    style_image,
):
    optimizer_patch, optimizer_mock = optimizer_mocks
    optim_loop_patch = default_transformer_optim_loop_patch

    paper.johnson_alahi_li_2016_training(image_loader, style_image)

    optimizer_patch.assert_called_once_with(transformer_mocks[1])
    optim_loop_patch.assert_called_once()

    _, kwargs = optim_loop_patch.call_args
    assert kwargs["optimizer"] is optimizer_mock
