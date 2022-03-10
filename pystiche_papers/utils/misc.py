import contextlib
import csv
import hashlib
import random
import shutil
import tempfile
from collections import OrderedDict
from copy import copy
from distutils.util import strtobool
from os import path
from typing import Any, Callable, Dict, Iterator, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Tuple, Union, cast

import numpy as np

import torch
from torch import hub, nn
from torch.hub import _get_torch_home
from torch.utils.data.dataloader import DataLoader

from pystiche.image import extract_batch_size, is_single_image, make_batched_image

__all__ = [
    "batch_up_image",
    "make_reproducible",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
    "load_state_dict_from_url",
    "str_to_bool",
    "load_urls_from_csv",
    "select_url_from_csv",
]


def batch_up_image(
    image: torch.Tensor,
    desired_batch_size: Optional[int] = None,
    loader: Optional[DataLoader] = None,
) -> torch.Tensor:
    def extract_batch_size_from_loader(loader: DataLoader) -> int:
        batch_size = loader.batch_size
        if batch_size is not None:
            return batch_size

        try:
            batch_size = loader.batch_sampler.batch_size  # type: ignore[union-attr]
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
) -> OrderedDict[str, torch.Tensor]:
    # This is just for compatibility with torch==1.6.0 until
    # https://github.com/pytorch/pytorch/issues/42596 is resolved
    if model_dir is None:
        model_dir = path.join(hub.get_dir(), "checkpoints")
    if file_name is None:
        file_name = path.basename(url)

    try:
        return cast(
            OrderedDictType[str, torch.Tensor],
            hub.load_state_dict_from_url(
                url, model_dir=model_dir, file_name=file_name, **kwargs
            ),
        )
    except RuntimeError as error:
        if str(error) != "Only one file(not dir) is allowed in the zipfile":
            raise error

        cached_file = path.join(model_dir, file_name)
        return cast(
            OrderedDictType[str, torch.Tensor],
            torch.load(cached_file, map_location=map_location),
        )


def str_to_bool(val: str) -> bool:
    return bool(strtobool(val))


def load_urls_from_csv(
    file: str,
    converters: Optional[Dict[str, Callable[[str], Any]]] = None,
    return_fieldnames: bool = False,
) -> Union[Dict[Tuple[Any, ...], str], Tuple[Dict[Tuple[Any, ...], str], List[str]]]:
    if converters is None:
        converters = {}
    with open(file, "r") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise RuntimeError(f"The file {file} is empty")
        fieldnames = list(copy(reader.fieldnames))
        if "url" not in fieldnames:
            raise RuntimeError(f"The file {file} does not contain an 'url' field")
        fieldnames.remove("url")

        for fieldname in fieldnames:
            if fieldname not in converters:
                converters[fieldname] = lambda x: x

        urls = {
            tuple(
                converters[field_name](row[field_name]) for field_name in fieldnames
            ): row["url"]
            for row in reader
        }

        if not return_fieldnames:
            return urls

        return urls, fieldnames


def select_url_from_csv(
    file: str,
    config: Tuple[Any, ...],
    converters: Optional[Dict[str, Callable[[str], Any]]] = None,
) -> str:
    urls, fieldnames = cast(
        Tuple[Dict[Tuple[Any, ...], str], List[str]],
        load_urls_from_csv(file, converters=converters, return_fieldnames=True),
    )
    try:
        return urls[config]
    except KeyError as error:
        msg = "No URL is available for the configuration:\n\n"
        msg += "\n".join(
            f"{fieldname}: {value}" for fieldname, value in zip(fieldnames, config)
        )
        raise RuntimeError(msg) from error
