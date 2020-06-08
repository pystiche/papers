import unittest
import torch

from pystiche_papers import utils

from .utils import skip_if_cuda_not_available


class TestMisc(unittest.TestCase):
    def test_get_sha256_hash(self):
        # FIXME
        # actual = utils.get_sha256_hash(self.default_image_file())
        # desired = "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683"
        # self.assertEqual(actual, desired)
        pass


class TestCuda(unittest.TestCase):
    @staticmethod
    def create_large_cuda_tensor(size_in_gb=256):
        return torch.empty((size_in_gb, *[1024] * 3), device="cuda", dtype=torch.uint8)

    @skip_if_cuda_not_available
    def test_use_cuda_out_of_memory_error(self):
        with self.assertRaises(utils.CudaOutOfMemoryError):
            with utils.use_cuda_out_of_memory_error():
                self.create_large_cuda_tensor()

    @skip_if_cuda_not_available
    def test_abort_if_cuda_memory_exausts(self):
        create_large_cuda_tensor = utils.abort_if_cuda_memory_exausts(
            self.create_large_cuda_tensor
        )

        with self.assertWarns(utils.CudaOutOfMemoryWarning):
            create_large_cuda_tensor()
