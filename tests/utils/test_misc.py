import pytest

import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from pystiche.image import extract_batch_size, make_single_image
from pystiche_papers import utils


def test_batch_up_image(input_image):
    batch_size = 3

    batched_up_image = utils.batch_up_image(input_image, batch_size)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_with_single_image(input_image):
    batch_size = 3

    batched_up_image = utils.batch_up_image(make_single_image(input_image), batch_size)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_with_batched_image(multi_input_image):
    with pytest.raises(RuntimeError):
        utils.batch_up_image(multi_input_image)


def test_batch_up_image_missing_arg(input_image):
    with pytest.raises(RuntimeError):
        utils.batch_up_image(input_image)


def test_batch_up_image_loader(input_image):
    batch_size = 3
    dataset = ()
    loader = DataLoader(dataset, batch_size=batch_size)

    batched_up_image = utils.batch_up_image(input_image, loader=loader)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_loader_with_batch_sampler(input_image):
    batch_size = 3
    dataset = ()
    batch_sampler = BatchSampler(
        SequentialSampler(dataset), batch_size, drop_last=False
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    batched_up_image = utils.batch_up_image(input_image, loader=loader)
    assert extract_batch_size(batched_up_image) == batch_size


def test_make_reproducible(subtests):
    seed = 123
    utils.make_reproducible(seed)

    try:
        import numpy as np

        with subtests.test(msg="numpy random seed"):
            numpy_seed = np.random.get_state()[1][0]
            assert numpy_seed == seed
    except ImportError:
        pass

    with subtests.test(msg="torch random seed"):
        torch_seed = torch.initial_seed()
        assert torch_seed == seed

    cudnn = torch.backends.cudnn
    if cudnn.is_available():
        with subtests.test(msg="cudnn state"):
            assert cudnn.deterministic
            assert not cudnn.benchmark


def test_make_reproducible_uint32_seed():
    seed = 123456789
    assert utils.make_reproducible(seed) == seed


def test_save_state_dict():
    pass
