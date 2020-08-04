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
