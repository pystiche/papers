import logging
import re
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from pystiche.image import extract_batch_size, make_single_image
from pystiche.optim import OptimLogger
from pystiche_papers import utils

from tests import assets
from tests import utils as utils_


def test_same_size_padding():
    assert utils.same_size_padding(kernel_size=1) == 0
    assert utils.same_size_padding(kernel_size=3) == 1
    assert utils.same_size_padding(kernel_size=(1, 3)) == (0, 1)


def test_same_size_output_padding():
    assert utils.same_size_output_padding(stride=1) == 0
    assert utils.same_size_output_padding(stride=2) == 1
    assert utils.same_size_output_padding(stride=(1, 2)) == (0, 1)


def test_is_valid_padding():
    assert utils.is_valid_padding(1)
    assert not utils.is_valid_padding(0)
    assert not utils.is_valid_padding(-1)

    assert utils.is_valid_padding((1, 2))
    assert not utils.is_valid_padding((1, 0, -1))


def test_paper_replication(subtests, caplog):
    optim_logger = OptimLogger()
    starting_offset = optim_logger._environ_level_offset
    with caplog.at_level(logging.INFO, optim_logger.logger.name):
        meta = {
            "title": "test_paper_replication",
            "url": "https://github.com/pmeier/pystiche_papers",
            "author": "pystiche_replication",
            "year": "2020",
        }
        with utils.paper_replication(optim_logger, **meta):
            with subtests.test("offset"):
                assert optim_logger._environ_level_offset == starting_offset + 1

            with subtests.test("logging level"):
                for record in caplog.records:
                    assert record.levelno == logging.INFO

            with subtests.test("text"):
                for value in meta.values():
                    assert value in caplog.text


def test_batch_up_image(image):
    batch_size = 3

    batched_up_image = utils.batch_up_image(image, batch_size)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_with_single_image(image):
    batch_size = 3

    batched_up_image = utils.batch_up_image(make_single_image(image), batch_size)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_with_batched_image(batch_image):
    with pytest.raises(RuntimeError):
        utils.batch_up_image(batch_image, 2)


def test_batch_up_image_missing_arg(image):
    with pytest.raises(RuntimeError):
        utils.batch_up_image(image)


def test_batch_up_image_loader(image):
    batch_size = 3
    dataset = ()
    loader = DataLoader(dataset, batch_size=batch_size)

    batched_up_image = utils.batch_up_image(image, loader=loader)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_loader_with_batch_sampler(image):
    batch_size = 3
    dataset = ()
    batch_sampler = BatchSampler(
        SequentialSampler(dataset), batch_size, drop_last=False
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    batched_up_image = utils.batch_up_image(image, loader=loader)
    assert extract_batch_size(batched_up_image) == batch_size


def test_batch_up_image_loader_with_batch_sampler_no_batch_size(subtests, image):
    class NoBatchSizeBatchSampler(BatchSampler):
        def __init__(self):
            pass

    class WrongTypeBatchSizeBatchSampler(BatchSampler):
        def __init__(self):
            self.batch_size = None

    with subtests.test("no batch_size"):
        loader = DataLoader((), batch_sampler=NoBatchSizeBatchSampler())
        with pytest.raises(RuntimeError):
            utils.batch_up_image(image, loader=loader)

    with subtests.test("wrong type batch_size"):
        loader = DataLoader((), batch_sampler=WrongTypeBatchSizeBatchSampler())
        with pytest.raises(RuntimeError):
            utils.batch_up_image(image, loader=loader)


def test_make_reproducible():
    def get_random_tensors():
        return torch.rand(10), torch.randn(10), torch.randint(10, (10,))

    utils.make_reproducible()
    tensors1 = get_random_tensors()

    utils.make_reproducible()
    tensors2 = get_random_tensors()
    tensors3 = get_random_tensors()

    ptu.assert_allclose(tensors1, tensors2)

    with pytest.raises(AssertionError):
        ptu.assert_allclose(tensors2, tensors3)


def test_make_reproducible_seeds(subtests, mocker):
    mocks = [
        (name, mocker.patch(f"pystiche_papers.utils.misc.{rel_import}"))
        for name, rel_import in (
            ("standard library", "random.seed"),
            ("numpy", "np.random.seed"),
            ("torch", "torch.manual_seed"),
        )
    ]

    seed = 123
    utils.make_reproducible(seed)

    for name, mock in mocks:
        assert mock.call_args[0][0] == seed


def test_make_reproducible_cudnn(mocker):
    cudnn_mock = mocker.patch("pystiche_papers.utils.misc.torch.backends.cudnn")
    cudnn_mock.is_available = lambda: True

    utils.make_reproducible()

    assert cudnn_mock.deterministic
    assert not cudnn_mock.benchmark


def test_make_reproducible_uint32_seed():
    uint32_max = 2 ** 32 - 1

    assert utils.make_reproducible(uint32_max) == uint32_max
    assert utils.make_reproducible(uint32_max + 1) == 0


def test_make_reproducible_no_standard_library(mocker):
    mock = mocker.patch("pystiche_papers.utils.misc.random.seed")
    utils.make_reproducible(seed_standard_library=False)

    assert not mock.called


def test_get_sha256_hash():
    file = path.join(assets.root(), "images", "small_0.png")
    actual = utils.get_sha256_hash(file)
    desired = "a2c1cc785a1b94eaa853ca13cc9314cc08f99ca2a746a3fa3a713a65dc2cfe05"
    assert actual == desired


@pytest.fixture
def conv2d_module():
    torch.manual_seed(0)
    return nn.Conv2d(3, 3, 1)


def test_save_state_dict(subtests, tmpdir, conv2d_module):
    state_dict = conv2d_module.state_dict()
    file = utils.save_state_dict(state_dict, "state_dict", root=tmpdir)
    ptu.assert_allclose(torch.load(file), state_dict)


def test_save_state_dict_module(subtests, tmpdir, conv2d_module):
    file = utils.save_state_dict(conv2d_module, "conv2d", root=tmpdir)
    ptu.assert_allclose(torch.load(file), conv2d_module.state_dict())


@utils_.skip_if_cuda_not_available
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
