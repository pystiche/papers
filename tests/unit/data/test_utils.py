import pytest

import torch

from pystiche_papers.data import utils

from tests import mocks


@pytest.fixture(scope="module")
def batch_sampler_factory():
    class BatchSampler(utils.NumIterationsBatchSampler):
        def infinite_iterator(self):
            while True:
                yield 0

    def factory(num_iterations, batch_size=1, drop_last=False):
        return BatchSampler(
            [None], num_iterations, batch_size=batch_size, drop_last=drop_last
        )

    return factory


def test_NumIterationsBatchSampler_len(batch_sampler_factory):
    num_iterations = 5
    batch_size = 2

    batch_sampler = batch_sampler_factory(num_iterations, batch_size=batch_size)

    assert len(batch_sampler) == 3
    assert len(tuple(iter(batch_sampler))) == 3


def test_NumIterationsBatchSampler_len_drop_last(batch_sampler_factory):
    num_iterations = 5
    batch_size = 2
    batch_sampler = batch_sampler_factory(
        num_iterations, batch_size=batch_size, drop_last=True
    )

    assert len(batch_sampler) == 2
    assert len(tuple(iter(batch_sampler))) == 2


def test_SequentialNumIterationsBatchSampler():
    data_source = [None] * 3
    num_iterations = 6
    batch_size = 2

    batch_sampler = utils.SequentialNumIterationsBatchSampler(
        data_source, num_iterations, batch_size=batch_size
    )

    actual = tuple(iter(batch_sampler))
    desired = ([0, 1], [2, 0], [1, 2])
    assert actual == desired


def test_RandomNumIterationsBatchSampler(mocker):
    def randint(high, size, *args, **kwargs):
        if size:
            raise pytest.UsageError
        return torch.tensor(high, dtype=torch.long)

    mocker.patch(
        mocks.make_mock_target("data", "utils", "torch", "randint"),
        side_effect=randint,
    )

    data_source = [None] * 3
    num_iterations = 6
    batch_size = 2

    batch_sampler = utils.RandomNumIterationsBatchSampler(
        data_source, num_iterations, batch_size=batch_size, drop_last=True
    )

    actual = tuple(iter(batch_sampler))
    expected = tuple(
        [len(data_source)] * batch_size for _ in range(num_iterations // batch_size)
    )
    assert actual == expected


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
