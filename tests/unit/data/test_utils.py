import pytorch_testing_utils as ptu
from torch import nn, optim

from pystiche_papers.data import utils


def test_InfiniteCycleBatchSampler():
    data_source = [None] * 3
    batch_size = 2

    batch_sampler = utils.InfiniteCycleBatchSampler(data_source, batch_size=batch_size)

    actual = []
    for idx, batch in enumerate(batch_sampler):
        if idx == 6:
            break
        actual.append(batch)
    actual = tuple(actual)

    desired = ((0, 1), (2, 0), (1, 2)) * 2
    assert actual == desired


def test_FiniteCycleBatchSampler():
    data_source = [None] * 3
    num_batches = 6
    batch_size = 2

    batch_sampler = utils.FiniteCycleBatchSampler(
        data_source, num_batches, batch_size=batch_size
    )

    actual = tuple(iter(batch_sampler))
    desired = ((0, 1), (2, 0), (1, 2)) * 2
    assert actual == desired


def test_InfiniteCycleBatchSampler_len():
    data_source = [None] * 3
    num_batches = 2
    batch_sampler = utils.FiniteCycleBatchSampler(data_source, num_batches)
    assert len(batch_sampler) == num_batches


def test_DelayedExponentialLR():
    base_lr = 1e3
    transformer = nn.Conv2d(3, 3, 1)
    gamma = 0.1
    delay = 2
    num_epochs = 5

    def get_optimizer(transformer):
        return optim.Adam(transformer.parameters(), lr=base_lr)

    optimizer = get_optimizer(transformer)
    lr_scheduler = utils.DelayedExponentialLR(optimizer, gamma, delay)

    for i in range(num_steps):
        if i >= delay:
            base_lr *= gamma

        param_group = optimizer.param_groups[0]
        assert param_group["lr"] == ptu.approx(base_lr)
        optimizer.step()
        lr_scheduler.step()
