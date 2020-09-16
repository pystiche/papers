import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn, optim

import pystiche_papers.sanakoyeu_et_al_2018 as paper


def test_preprocessor(input_image):
    preprocessor = paper.preprocessor()
    assert isinstance(preprocessor, nn.Module)

    actual = preprocessor(input_image)
    expected = input_image * 2 - 1
    ptu.assert_allclose(actual, expected)


def test_postprocessor(input_image):
    postprocessor = paper.postprocessor()
    assert isinstance(postprocessor, nn.Module)

    actual = postprocessor(input_image)
    expected = (input_image + 1) / 2
    ptu.assert_allclose(actual, expected)


def test_optimizer_modules(subtests):
    transformer = nn.Conv2d(3, 3, 1)
    params = tuple(transformer.parameters())

    for optimizer_params in [transformer, transformer.parameters()]:
        with subtests.test("parameters"):
            optimizer = paper.optimizer(optimizer_params)

            assert isinstance(optimizer, optim.Adam)
            assert len(optimizer.param_groups) == 1

            param_group = optimizer.param_groups[0]

            with subtests.test(msg="optimization params"):
                assert len(param_group["params"]) == len(params)
                for actual, desired in zip(param_group["params"], params):
                    assert actual is desired

            with subtests.test(msg="optimizer properties"):
                assert param_group["lr"] == ptu.approx(2e-4)


def test_lr_scheduler():
    transformer = nn.Conv2d(3, 3, 1)
    optimizer = paper.optimizer(transformer)
    lr_scheduler = paper.lr_scheduler(optimizer)

    assert isinstance(lr_scheduler, paper.DelayedExponentialLR)


def test_ExponentialMovingAverageMeter(subtests):
    vals = [0.6, 0.7, 0.8]
    init_val = 0.8
    smoothing_factor = 0.05
    name = "test_ema"
    ema = paper.ExponentialMovingAverageMeter(name, init_val)

    with subtests.test("name"):
        assert ema.name == name

    with subtests.test("global_avg"):
        desired = init_val
        for val in vals:
            ema.update(val)
            desired = desired * (1.0 - smoothing_factor) + smoothing_factor * val

        current = ema.global_avg
        assert current == ptu.approx(desired)


def test_ExponentialMovingAverageMeter_update_with_tensor(subtests):
    vals = [torch.tensor([0.6]), torch.tensor([0.7]), torch.tensor([0.8])]
    init_val = 0.8
    smoothing_factor = 0.05
    name = "test_ema"
    ema = paper.ExponentialMovingAverageMeter(name, init_val)

    with subtests.test("init_val"):
        assert ema.global_avg == pytest.approx(init_val)

    with subtests.test("global_avg"):
        desired = init_val
        for val in vals:
            ema.update(val)
            desired = desired * (1.0 - smoothing_factor) + smoothing_factor * val

        current = ema.global_avg
        assert current == ptu.approx(desired)


def test_ExponentialMovingAverageMeter_str_smoke():
    init_val = 0.8
    meter = paper.ExponentialMovingAverageMeter(
        "test_exponential_moving_average_meter", init_val
    )
    assert isinstance(str(meter), str)
