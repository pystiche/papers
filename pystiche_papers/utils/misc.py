import contextlib
import hashlib
import random
import shutil
import tempfile
from collections import OrderedDict
from os import path
from typing import Any, Dict, Iterator, Optional, Union, cast

import numpy as np

import torch
from torch import hub, nn
from torch.hub import _get_torch_home
from torch.utils.data.dataloader import DataLoader

from pystiche.image import extract_batch_size, is_single_image, make_batched_image
from pystiche.optim import OptimLogger

__all__ = [
    "batch_up_image",
    "paper_replication",
    "make_reproducible",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
    "load_state_dict_from_url",
]


def batch_up_image(
    image: torch.Tensor,
    desired_batch_size: Optional[int] = None,
    loader: Optional[DataLoader] = None,
) -> torch.Tensor:
    def extract_batch_size_from_loader(loader: DataLoader) -> int:
        batch_size = cast(Optional[int], loader.batch_size)
        if batch_size is not None:
            return batch_size

        try:
            batch_size = loader.batch_sampler.batch_size  # type: ignore[attr-defined]
            assert isinstance(batch_size, int)
            return batch_size
        except (AttributeError, AssertionError):
            raise RuntimeError

    if desired_batch_size is None and loader is None:
        raise RuntimeError

    if desired_batch_size is None:
        desired_batch_size = extract_batch_size_from_loader(cast(DataLoader, loader))

    if is_single_image(image):
        image = make_batched_image(image)
    elif extract_batch_size(image) > 1:
        raise RuntimeError

    return image.repeat(desired_batch_size, 1, 1, 1)


@contextlib.contextmanager
def paper_replication(
    optim_logger: OptimLogger, title: str, url: str, author: str, year: Union[str, int]
) -> Iterator:
    header = "\n".join(
        (
            "Replication of the paper",
            f"'{title}'",
            url,
            "authored by",
            author,
            f"in {str(year)}",
        )
    )
    with optim_logger.environment(header):
        yield


def make_reproducible(
    seed: Optional[Any] = 0, seed_standard_library: bool = True
) -> int:
    def maybe_seed_standard_library(seed: int) -> None:
        if seed_standard_library:
            random.seed(seed)

    def seed_numpy(seed: int) -> None:
        np.random.seed(seed)

    def seed_torch(seed: int) -> None:
        torch.manual_seed(seed)

    def maybe_set_cudnn() -> None:
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # the numpy random generator only accepts uint32 values
    seed = hash(seed) % 2 ** 32

    maybe_seed_standard_library(seed)
    seed_numpy(seed)
    seed_torch(seed)
    maybe_set_cudnn()

    return seed


@contextlib.contextmanager
def get_tmp_dir(**mkdtemp_kwargs: Any) -> Iterator[str]:
    tmp_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


def get_sha256_hash(file: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_state_dict(
    input: Union[Dict[str, torch.Tensor], nn.Module],
    name: str,
    root: Optional[str] = None,
    to_cpu: bool = True,
    hash_len: int = 8,
    ext: str = ".pth",
) -> str:
    if isinstance(input, nn.Module):
        state_dict = input.state_dict()
    else:
        state_dict = OrderedDict(input)

    if to_cpu:
        state_dict = OrderedDict(
            [(key, tensor.cpu()) for key, tensor in state_dict.items()]
        )

    if root is None:
        root = _get_torch_home()

    with get_tmp_dir() as tmp_dir:
        tmp_file = path.join(tmp_dir, "tmp")
        torch.save(state_dict, tmp_file)
        sha256 = get_sha256_hash(tmp_file)

        file = path.join(root, f"{name}-{sha256[:hash_len]}{ext}")
        shutil.move(tmp_file, file)

    return file


def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: Optional[Union[torch.device, str]] = None,
    file_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    # This is just for compatibility with torch==1.6.0 until
    # https://github.com/pytorch/pytorch/issues/42596 is resolved
    if model_dir is None:
        model_dir = path.join(hub.get_dir(), "checkpoints")
    if file_name is None:
        file_name = path.basename(url)

    try:
        return cast(
            Dict[str, torch.Tensor],
            hub.load_state_dict_from_url(
                url, model_dir=model_dir, file_name=file_name, **kwargs
            ),
        )
    except RuntimeError as error:
        if str(error) != "Only one file(not dir) is allowed in the zipfile":
            raise error

        cached_file = path.join(model_dir, file_name)
        return cast(
            Dict[str, torch.Tensor], torch.load(cached_file, map_location=map_location)
        )
