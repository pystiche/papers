import pytest

import torch

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope="session", autouse=True)
def cuda_init():
    if CUDA_AVAILABLE:
        torch.empty(1, device="cuda")


@pytest.fixture(scope="session")
def cuda_device():
    if not CUDA_AVAILABLE:
        raise RuntimeError

    ordinal = torch.cuda.current_device()
    return torch.device("cuda", ordinal)
