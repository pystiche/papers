import unittest

from pystiche_papers.data import utils


class TestUtils(unittest.TestCase):
    def test_InfiniteCycleBatchSampler(self):
        data_source = [None] * 3
        batch_size = 2

        batch_sampler = utils.InfiniteCycleBatchSampler(
            data_source, batch_size=batch_size
        )

        actual = []
        for idx, batch in enumerate(batch_sampler):
            if idx == 6:
                break
            actual.append(batch)
        actual = tuple(actual)

        desired = ((0, 1), (2, 0), (1, 2)) * 2
        self.assertEqual(actual, desired)

    def test_FiniteCycleBatchSampler(self):
        data_source = [None] * 3
        num_batches = 6
        batch_size = 2

        batch_sampler = utils.FiniteCycleBatchSampler(
            data_source, num_batches, batch_size=batch_size
        )

        actual = tuple(iter(batch_sampler))
        desired = ((0, 1), (2, 0), (1, 2)) * 2
        self.assertEqual(actual, desired)

    def test_InfiniteCycleBatchSampler_len(self):
        data_source = [None] * 3
        num_batches = 2
        batch_sampler = utils.FiniteCycleBatchSampler(data_source, num_batches)
        self.assertEqual(len(batch_sampler), num_batches)
