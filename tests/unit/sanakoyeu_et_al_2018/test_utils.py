import pytorch_testing_utils as ptu
from torch import nn, optim

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import ops


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


def test_ExponentialMovingAverage(subtests):
    vals = [0.6, 0.7, 0.8]
    init_val = 0.8
    smoothing_factor = 0.05
    name = "test_ema"
    ema = paper.ExponentialMovingAverage(name)

    with subtests.test("name"):
        assert ema.name == name

    with subtests.test("local average"):
        desired = init_val
        for val in vals:
            ema.update(val)
            desired = desired * (1.0 - smoothing_factor) + smoothing_factor * val

        current = ema.local_avg()
        assert current == ptu.approx(desired)


def test_class_ContentOperatorContainer_set_target_image(target_image):
    class RegularizationTestOperator(ops.PixelRegularizationOperator):
        def input_image_to_repr(self, image):
            pass

        def calculate_score(self, input_repr):
            pass

    class ComparisonTestOperator(ops.PixelComparisonOperator):
        def target_image_to_repr(self, image):
            return image, None

        def input_image_to_repr(self, image, ctx):
            pass

        def calculate_score(self, input_repr, target_repr, ctx):
            pass

    def get_container():
        return paper.ContentOperatorContainer(
            (
                ("regularization", RegularizationTestOperator()),
                ("comparison", ComparisonTestOperator()),
            )
        )

    container = get_container()
    container.set_target_image(target_image)

    actual = container.comparison.target_image
    desired = target_image
    ptu.assert_allclose(actual, desired)


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
