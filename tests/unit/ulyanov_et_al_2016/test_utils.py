import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

from .utils import impl_params_and_instance_norm


def test_hyper_parameters_smoke():
    hyper_parameters = paper.hyper_parameters()
    assert isinstance(hyper_parameters, HyperParameters)


@impl_params_and_instance_norm
def test_hyper_parameters_content_loss(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "content_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layer == "relu4_2"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(
            6e-1 if impl_params and not instance_norm else 1e0
        )


@impl_params_and_instance_norm
def test_hyper_parameters_style_loss(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "style_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layers == (
            ("relu1_1", "relu2_1", "relu3_1", "relu4_1")
            if impl_params and instance_norm
            else ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
        )

    with subtests.test("layer_weights"):
        assert hyper_parameters.layer_weights == [1e3] * 4 if impl_params and instance_norm else [1e3] * 5

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(1e0)


@impl_params_and_instance_norm
def test_hyper_parameters_content_transform(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "content_transform"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_size"):
        assert hyper_parameters.edge_size == 256


@impl_params_and_instance_norm
def test_hyper_parameters_style_transform(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "style_transform"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("edge_size"):
        assert hyper_parameters.edge_size == 256

    with subtests.test("edge"):
        assert hyper_parameters.edge == "long"

    with subtests.test("interpolation_mode"):
        assert hyper_parameters.interpolation_mode == (
            "bicubic" if impl_params and instance_norm else "bilinear"
        )


@impl_params_and_instance_norm
def test_hyper_parameters_batch_sampler(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "batch_sampler"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("batch_size"):
        assert hyper_parameters.batch_size == (
            (1 if instance_norm else 4) if impl_params else 16
        )


@impl_params_and_instance_norm
def test_hyper_parameters_optimizer(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "optimizer"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("lr"):
        assert hyper_parameters.lr == 1e-3 if impl_params and instance_norm else 1e-1


@impl_params_and_instance_norm
def test_hyper_parameters_lr_scheduler(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    sub_params = "lr_scheduler"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("lr_decay"):
        assert hyper_parameters.lr_decay == 0.8 if impl_params else 0.7


@impl_params_and_instance_norm
def test_hyper_parameters_num_images(impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    num_batches_per_epoch = hyper_parameters.batch_sampler.num_batches
    num_epochs = hyper_parameters.num_epochs

    num_images = num_batches_per_epoch * num_epochs

    assert num_images == (50_000 if instance_norm else 3_000) if impl_params else 2_000


@impl_params_and_instance_norm
def test_hyper_parameters_lr_decay_delay(impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    )

    num_batches = hyper_parameters.batch_sampler.num_batches
    delay = hyper_parameters.lr_scheduler.delay

    num_batches_before_first_decay = num_batches * (delay + 1)
    num_batches_between_decays = num_batches

    assert (
        num_batches_before_first_decay == (2_000 if instance_norm else 300)
        if impl_params
        else 1000
    )
    assert (
        num_batches_between_decays == (2_000 if instance_norm else 300)
        if impl_params
        else 200
    )


@pytest.mark.slow
def test_ulyanov_et_al_2016_multi_layer_encoder(subtests):
    multi_layer_encoder = paper.multi_layer_encoder()
    assert isinstance(multi_layer_encoder, enc.VGGMultiLayerEncoder)

    with subtests.test("internal preprocessing"):
        assert "preprocessing" not in multi_layer_encoder

    with subtests.test("inplace"):
        relu_modules = [
            module
            for module in multi_layer_encoder.modules()
            if isinstance(module, nn.ReLU)
        ]
        assert all(module.inplace for module in relu_modules)


def test_ulyanov_et_al_2016_preprocessor():
    assert isinstance(paper.preprocessor(), transforms.CaffePreprocessing)


def test_ulyanov_et_al_2016_postprocessor():
    assert isinstance(paper.postprocessor(), transforms.CaffePostprocessing)


@impl_params_and_instance_norm
def test_ulyanov_et_al_2016_optimizer(subtests, impl_params, instance_norm):
    transformer = nn.Conv2d(3, 3, 1)
    params = tuple(transformer.parameters())

    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).optimizer

    optimizer = paper.optimizer(
        transformer, impl_params=impl_params, instance_norm=instance_norm
    )

    assert isinstance(optimizer, optim.Adam)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == len(params)
        for actual, desired in zip(param_group["params"], params):
            assert actual is desired

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(hyper_parameters.lr)


def test_DelayedExponentialLR():
    transformer = nn.Conv2d(3, 3, 1)
    gamma = 0.1
    delay = 2
    num_steps = 5
    optimizer = paper.optimizer(transformer)
    lr_scheduler = paper.DelayedExponentialLR(optimizer, gamma, delay)

    param_group = optimizer.param_groups[0]
    base_lr = param_group["lr"]
    for i in range(num_steps):
        if i >= delay:
            base_lr *= gamma

        param_group = optimizer.param_groups[0]
        assert param_group["lr"] == ptu.approx(base_lr)
        optimizer.step()
        lr_scheduler.step()


@impl_params_and_instance_norm
def test_ulyanov_et_al_2016_lr_scheduler(subtests, impl_params, instance_norm):
    transformer = nn.Conv2d(3, 3, 1)
    optimizer = paper.optimizer(transformer)

    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).lr_scheduler

    lr_scheduler = paper.lr_scheduler(
        optimizer, impl_params=impl_params, instance_norm=instance_norm
    )

    assert isinstance(lr_scheduler, paper.DelayedExponentialLR)

    with subtests.test("lr_decay"):
        assert lr_scheduler.gamma == hyper_parameters.lr_decay

    with subtests.test("delay"):
        assert lr_scheduler.delay == hyper_parameters.delay
