import contextlib
import os
import shutil
import tempfile

import pytest

import torch

from pystiche import image

__all__ = ["get_tempdir", "skip_if_cuda_not_available", "is_callable", "create_guides"]


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
def get_tempdir(**mkdtemp_kwargs):
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, onerror=onerror)


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
