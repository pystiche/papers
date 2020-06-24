import torch

from pystiche_papers import utils


def test_batch_up_image():
    pass


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
