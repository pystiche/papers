import contextlib
import copy
import itertools
import os
import shutil
import tempfile
from os import path

import pytest

import pytorch_testing_utils as ptu
import torch

from pystiche import image

__all__ = [
    "rmtree",
    "get_tempdir",
    "skip_if_cuda_not_available",
    "is_callable",
    "create_guides",
    "call_args_list_to_dict",
    "generate_param_combinations",
    "call_args_to_kwargs_only",
]


# Copied from
# https://pypi.org/project/pathutils/
def onerror(func, path, exc_info):
    """Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat

    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def rmtree(dir):
    if path.exists(dir):
        shutil.rmtree(dir, onerror=onerror)


@contextlib.contextmanager
def get_tempdir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        rmtree(tmp_dir)


skip_if_cuda_not_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available."
)


def is_callable(obj):
    return hasattr(obj, "__call__")


def create_guides(img):
    height, width = image.extract_image_size(img)
    top_height = height // 2
    bottom_height = height - top_height
    top_mask = torch.cat(
        (
            torch.ones([1, 1, top_height, width], dtype=torch.bool),
            torch.zeros([1, 1, bottom_height, width], dtype=torch.bool),
        ),
        2,
    )
    bottom_mask = ~top_mask
    return {"top": top_mask.float(), "bottom": bottom_mask.float()}


def _allclose(*args, **kwargs):
    try:
        ptu.assert_allclose(*args, **kwargs)
        return True
    except AssertionError:
        return False


def _xnor(a, b):
    return not (a ^ b)


def call_args_list_to_dict(call_args_list, map, args_idx=None, kwargs_key=None):
    if _xnor(args_idx is None, kwargs_key is None):
        raise pytest.UsageError
    if args_idx is not None:

        def check(call_args, tensor):
            args, kwargs = call_args
            return _allclose(args[args_idx], tensor)

    else:  # kwargs_key is not None

        def check(call_args, tensor):
            args, kwargs = call_args
            return _allclose(kwargs[kwargs_key], tensor)

    call_args_dict = {}
    processed_idcs = []
    for name, tensor in map.items():
        for idx, call_args in enumerate(call_args_list):
            if idx in processed_idcs:
                continue

            if check(call_args, tensor):
                call_args_dict[name] = call_args
                processed_idcs.append(idx)

    if len(call_args_dict) < len(call_args_list):
        raise pytest.UsageError

    return call_args_dict


def generate_param_combinations(**kwargs):
    names = tuple(kwargs.keys())
    iterables = tuple(kwargs.values())
    for params in itertools.product(*iterables):
        yield dict(zip(names, params))


def call_args_to_kwargs_only(call_args, *arg_names):
    if call_args is None:
        raise pytest.UsageError
    args, kwargs = call_args

    kwargs_only = copy.copy(kwargs)
    kwargs_only.update(dict(zip(arg_names, args)))
    return kwargs_only
