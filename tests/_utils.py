import contextlib
import os
import shutil
import tempfile
from importlib import util
from os import path

import pytest

import torch

from pystiche import image

__all__ = ["get_tmp_dir", "skip_if_cuda_not_available", "load_module", "is_callable"]


# Copied from
# https://pypi.org/project/pathutils/
def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

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


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, onerror=onerror)


skip_if_cuda_not_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available."
)


def load_module(location):
    name, ext = path.splitext(path.basename(location))
    is_package = ext != ".py"
    if is_package:
        location = path.join(location, "__init__.py")

    spec = util.spec_from_file_location(name, location=location)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_callable(obj):
    return hasattr(obj, "__call__")


def create_guides(img):
    height, width = image.extract_image_size(img)
    top_height = height // 2  # built in floor
    bottom_height = height - top_height
    mask = torch.stack(
        (
            torch.ones([1, 1, top_height, width], dtype=torch.bool),
            torch.zeros([1, 1, bottom_height, width], dtype=torch.bool),
        ),
        2,
    )
    return {"top": mask.float(), "bottom": (~mask).float()}
