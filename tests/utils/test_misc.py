import re
from os import path

import pytest

import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from pystiche.image import extract_batch_size, make_single_image
from pystiche_papers import utils

from .._utils import skip_if_cuda_not_available


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


@pytest.mark.skip(reason="No fixed asset is available to test this.")
def test_get_sha256_hash():
    raise RuntimeError


@pytest.fixture
def conv2d_module():
    torch.manual_seed(0)
    return nn.Conv2d(3, 3, 1)


def test_save_state_dict(subtests, tmpdir, conv2d_module):
    state_dict = conv2d_module.state_dict()

    file = utils.save_state_dict(state_dict, "state_dict", root=tmpdir)
    actual_state_dict = torch.load(file)

    assert actual_state_dict.keys() == state_dict.keys()
    for param_name, param in actual_state_dict.items():
        with subtests.test(param_name=param_name):
            # FIXME: approx
            assert torch.all(
                actual_state_dict[param_name] == state_dict[param_name]
            ).item()


def test_save_state_dict_module(subtests, tmpdir, conv2d_module):
    state_dict = conv2d_module.state_dict()

    file = utils.save_state_dict(conv2d_module, "conv2d", root=tmpdir)
    actual_state_dict = torch.load(file)

    assert actual_state_dict.keys() == state_dict.keys()
    for param_name, param in actual_state_dict.items():
        with subtests.test(param_name=param_name):
            # FIXME: approx
            assert torch.all(
                actual_state_dict[param_name] == state_dict[param_name]
            ).item()


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError

    ordinal = torch.cuda.current_device()
    return torch.device("cuda", ordinal)


@skip_if_cuda_not_available
def test_save_state_dict_not_to_cpu(subtests, tmpdir, conv2d_module, cuda_device):
    module = conv2d_module.to(cuda_device)

    file = utils.save_state_dict(module, "cuda", root=tmpdir, to_cpu=False)
    state_dict = torch.load(file)

    for name, param in state_dict.items():
        with subtests.test(name=name):
            assert param.device == cuda_device


def test_save_state_dict_file(subtests, tmpdir, conv2d_module):
    name = "file"
    ext = ".pp"
    hash_len = 12

    file = utils.save_state_dict(
        conv2d_module, name, root=tmpdir, hash_len=hash_len, ext=ext
    )

    match = re.match(
        fr"^{name}-(?P<hash>[0-9a-f]{{{hash_len}}})[.]{ext[1:]}$", path.basename(file)
    )
    assert match is not None

    hash = match.group("hash")
    assert hash == utils.get_sha256_hash(file)[:hash_len]
