import pytest

import torch

from pystiche_papers import utils


@pytest.fixture
def create_too_large_cuda_tensor():
    if torch.cuda.is_available():

        def creator():
            device_idx = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(device_idx)
            max_memory_in_bytes = device_properties.total_memory
            max_memory_in_gibibytes = max_memory_in_bytes / 1024 ** 3
            requested_memory_in_gibibytes = int(2 * max_memory_in_gibibytes)
            size = (requested_memory_in_gibibytes, *[1024] * 3)
            return torch.empty(
                size, device=torch.device("cuda", device_idx), dtype=torch.uint8
            )

    else:

        def creator():
            raise RuntimeError("CUDA out of memory")

    return creator


def test_use_cuda_out_of_memory_error(create_too_large_cuda_tensor):
    with pytest.raises(utils.CudaOutOfMemoryError):
        with utils.use_cuda_out_of_memory_error():
            create_too_large_cuda_tensor()


def test_abort_if_cuda_memory_exausts(create_too_large_cuda_tensor):
    create_too_large_cuda_tensor = utils.abort_if_cuda_memory_exausts(
        create_too_large_cuda_tensor
    )

    with pytest.warns(utils.CudaOutOfMemoryWarning):
        create_too_large_cuda_tensor()
