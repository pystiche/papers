import unittest.mock

import pytest

import pytorch_testing_utils as ptu
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import misc

from tests.utils import is_callable


def make_patch_target(name):
    return ".".join(("pystiche_papers", "sanakoyeu_et_al_2018", "_nst", name))


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
def optimizer_mocks(mocker, patcher):
    mock = mocker.Mock()
    patch = patcher("optimizer", return_value=mock)
    return patch, mock


@pytest.fixture
def lr_scheduler_mocks(mocker, patcher):
    mock = mocker.Mock()
    patch = patcher("_lr_scheduler", return_value=mock)
    return patch, mock


@pytest.fixture
def transformer_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("_transformer", return_value=mock)
    return patch, mock


@pytest.fixture
def discriminator_operator_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock(identity=True)
    patch = patcher("_discriminator_operator", return_value=mock)
    return patch, mock


@pytest.fixture
def transformer_loss_mocks(make_nn_module_mock, patcher):
    mock = make_nn_module_mock()
    attach_method_mock(mock, "set_content_image", return_value=None)
    attach_method_mock(mock, "set_style_image", return_value=None)
    patch = patcher("perceptual_loss", return_value=mock)
    return patch, mock


@pytest.fixture
def epoch_gan_optim_loop_patch(patcher):
    def side_effect(_, transformer, *args, **kwargs):
        return transformer

    return patcher("epoch_gan_optim_loop", prefix=False, side_effect=side_effect,)


@pytest.fixture
def image_loader(image):
    return DataLoader(TensorDataset(image), batch_size=2)


def reset_mocks(*mocks):
    for mock in mocks:
        mock.reset_mock()


@pytest.fixture
def training(epoch_gan_optim_loop_patch, image_loader, style_image):
    def training_(image_loader_=None, **kwargs):
        if image_loader_ is None:
            image_loader_ = image_loader
        output = paper.training(image_loader_, image_loader_, **kwargs)

        epoch_gan_optim_loop_patch.assert_called_once()
        args, kwargs = epoch_gan_optim_loop_patch.call_args

        return args, kwargs, output

    return training_


@pytest.mark.slow
def test_training_smoke(subtests, training, image_loader):
    args, kwargs, output = training(image_loader)
    (
        content_image_loader,
        style_image_loader,
        transformer,
        num_epochs,
        discriminator_criterion,
        transformer_criterion,
        transformer_criterion_update_fn,
    ) = args
    discriminator_lr_scheduler = kwargs["discriminator_lr_scheduler"]
    transformer_lr_scheduler = kwargs["transformer_lr_scheduler"]

    with subtests.test("content_image_loader"):
        assert content_image_loader is image_loader

    with subtests.test("style_image_loader"):
        assert style_image_loader is image_loader

    with subtests.test("transformer"):
        assert isinstance(transformer, type(paper.transformer()))

    with subtests.test("num_epochs"):
        assert isinstance(num_epochs, int)

    with subtests.test("discriminator_criterion"):
        assert isinstance(discriminator_criterion, type(paper.DiscriminatorLoss()))

    with subtests.test("transformer_criterion"):
        assert isinstance(transformer_criterion, type(paper.transformer_loss()()))

    with subtests.test("transformer_criterion_update_fn"):
        assert is_callable(transformer_criterion_update_fn)

    with subtests.test("discriminator_lr_scheduler"):
        assert isinstance(discriminator_lr_scheduler, ExponentialLR)
        assert isinstance(
            discriminator_lr_scheduler.optimizer,
            type(paper.optimizer(discriminator_criterion)),
        )

    with subtests.test("transformer_lr_scheduler"):
        assert isinstance(transformer_lr_scheduler, ExponentialLR)
        assert isinstance(
            transformer_lr_scheduler.optimizer, type(paper.optimizer(transformer))
        )

    with subtests.test("output"):
        assert output is transformer


def test_training_device(
    subtests,
    optimizer_mocks,
    lr_scheduler_mocks,
    transformer_mocks,
    discriminator_operator_mocks,
    discriminator_loss_mocks,
    transformer_loss_mocks,
    training,
    style_image,
):
    training(style_image_=style_image)

    for mocks in (
        transformer_mocks,
        discriminator_loss_mocks,
        transformer_loss_mocks,
    ):
        _, mock = mocks
        mock = mock.to
        with subtests.test(mock.name):
            mock.assert_called_once_with(misc.get_device())


def test_training_transformer_train(
    optimizer_mocks,
    lr_scheduler_mocks,
    transformer_mocks,
    discriminator_operator_mocks,
    discriminator_loss_mocks,
    transformer_loss_mocks,
    training,
):
    args, _, _ = training()
    transformer = args[2]

    transformer.train.assert_called_once_with()


def test_training_criterion_eval(
    subtests,
    optimizer_mocks,
    lr_scheduler_mocks,
    transformer_mocks,
    discriminator_operator_mocks,
    discriminator_loss_mocks,
    transformer_loss_mocks,
    training,
):
    args, _, _ = training()
    discriminator_criterion = args[4]
    transformer_criterion = args[5]
    with subtests.test("discriminator_criterion"):
        discriminator_criterion.eval.assert_called_once_with()

    with subtests.test("transformer_criterion"):
        transformer_criterion.eval.assert_called_once_with()


def test_training_num_epochs(
    subtests,
    optimizer_mocks,
    lr_scheduler_mocks,
    transformer_mocks,
    discriminator_operator_mocks,
    discriminator_loss_mocks,
    transformer_loss_mocks,
    training,
    default_transformer_epoch_optim_loop_patch,
):
    mocks = (
        lr_scheduler_mocks[0],
        transformer_mocks[0],
        transformer_loss_mocks[0],
    )
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            reset_mocks(*mocks, default_transformer_epoch_optim_loop_patch)
            args, _, _ = training(impl_params=impl_params)
            num_epochs = args[3]

            assert num_epochs == 1 if impl_params else 3


def test_training_criterion_update_fn(
    optimizer_mocks,
    lr_scheduler_mocks,
    transformer_mocks,
    discriminator_operator_mocks,
    discriminator_loss_mocks,
    transformer_loss_mocks,
    training,
    content_image,
):
    args, _, _ = training()
    transformer_criterion = args[5]
    transformer_criterion_update_fn = args[6]

    assert not transformer_criterion.content_loss.has_target_image

    transformer_criterion_update_fn(content_image, transformer_criterion)
    assert transformer_criterion.content_loss.has_target_image
    ptu.assert_allclose(transformer_criterion.content_loss.target_image, content_image)


def test_training_transformer_lr_scheduler_optimizer(
    transformer_mocks, training,
):
    parameter = torch.empty(1)
    _, transformer_mock = transformer_mocks
    attach_method_mock(transformer_mock, "parameters", return_value=iter((parameter,)))

    _, kwargs, _ = training()
    lr_scheduler = kwargs["transformer_lr_scheduler"]
    optimizer = lr_scheduler.optimizer

    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 1

    param = param_group["params"][0]
    assert param is parameter


def test_training_discriminator_lr_scheduler_optimizer(
    discriminator_operator_mocks, training,
):
    parameter = torch.empty(1)
    _, discriminator_operator = discriminator_operator_mocks
    attach_method_mock(
        discriminator_operator, "parameters", return_value=iter((parameter,))
    )

    _, kwargs, _ = training()
    lr_scheduler = kwargs["discriminator_lr_scheduler"]
    optimizer = lr_scheduler.optimizer

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
    subtests, transformer_mocks, stylization, input_image,
):
    stylization(input_image)

    _, mock = transformer_mocks
    mock = mock.to
    with subtests.test(mock.name):
        mock.assert_called_once_with(input_image.device)


def test_stylization_transformer_eval(
    subtests, transformer_mocks, stylization, input_image,
):
    _, transformer = transformer_mocks
    reset_mocks(transformer)
    stylization(input_image)
    transformer.eval.assert_called_once_with()
