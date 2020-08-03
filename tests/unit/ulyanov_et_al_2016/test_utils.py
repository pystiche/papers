import pytest

import pytorch_testing_utils as ptu
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import enc
from pystiche.image import transforms


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


def test_ulyanov_et_al_2016_optimizer(subtests):
    transformer = nn.Conv2d(3, 3, 1)
    params = tuple(transformer.parameters())

    configs = (
        (True, True, 1e-3),
        (True, False, 1e-1),
        (False, True, 1e-1),
        (False, False, 1e-1),
    )
    for impl_params, instance_norm, lr in configs:
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
            assert param_group["lr"] == ptu.approx(lr)


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
        lr_scheduler.step()


def test_ulyanov_et_al_2016_lr_scheduler():
    transformer = nn.Conv2d(3, 3, 1)
    optimizer = paper.optimizer(transformer)
    for impl_params in (True, False):
        lr_scheduler = paper.lr_scheduler(optimizer, impl_params=impl_params)

        assert isinstance(
            lr_scheduler, ExponentialLR if impl_params else paper.DelayedExponentialLR,
        )
