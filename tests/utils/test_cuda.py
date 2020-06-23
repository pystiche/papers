import pytest

import torch

from pystiche_papers import utils

from .._utils import skip_if_cuda_not_available


@pytest.fixture(scope="module")
def create_large_cuda_tensor():
    size_in_gb = 256
    size = (size_in_gb, *[1024] * 3)
    return lambda: torch.empty(size, device="cuda", dtype=torch.uint8)


@skip_if_cuda_not_available
def test_use_cuda_out_of_memory_error(create_large_cuda_tensor):
    with pytest.raises(utils.CudaOutOfMemoryError):
        with utils.use_cuda_out_of_memory_error():
            create_large_cuda_tensor()


@skip_if_cuda_not_available
def test_abort_if_cuda_memory_exausts(create_large_cuda_tensor):
    create_large_cuda_tensor = utils.abort_if_cuda_memory_exausts(
        create_large_cuda_tensor
    )

    with pytest.warns(utils.CudaOutOfMemoryWarning):
        create_large_cuda_tensor()
