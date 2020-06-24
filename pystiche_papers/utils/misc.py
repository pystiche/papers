import contextlib
import hashlib
import random
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from os import path
from typing import Any, Callable, Dict, Iterator, Optional
from typing import Sequence as SequenceType
from typing import Tuple, TypeVar, Union, cast, overload

import torch
from torch import nn
from torch.hub import _get_torch_home
from torch.utils.data.dataloader import DataLoader

from pystiche.image import extract_batch_size, is_single_image, make_batched_image
from pystiche.optim import OptimLogger

__all__ = [
    "same_size_padding",
    "same_size_output_padding",
    "is_valid_padding",
    "batch_up_image",
    "paper_replication",
    "make_reproducible",
    "get_tmp_dir",
    "get_sha256_hash",
    "save_state_dict",
]

In = TypeVar("In")
Out = TypeVar("Out")


@overload
def elementwise(fn: Callable[[In], Out], inputs: In) -> Out:  # type: ignore[misc]
    ...


@overload
def elementwise(fn: Callable[[In], Out], inputs: SequenceType[In]) -> Tuple[Out, ...]:
    ...


def elementwise(
    fn: Callable[[In], Out], inputs: Union[In, SequenceType[In]]
) -> Union[Out, Tuple[Out, ...]]:
    if isinstance(inputs, Sequence):
        return tuple(fn(input) for input in inputs)
    return fn(inputs)


@overload
def same_size_padding(kernel_size: int) -> int:
    ...


@overload
def same_size_padding(kernel_size: SequenceType[int]) -> Tuple[int, ...]:
    ...


def same_size_padding(
    kernel_size: Union[int, SequenceType[int]]
) -> Union[int, Tuple[int, ...]]:
    return elementwise(lambda x: (x - 1) // 2, kernel_size)  # type: ignore[no-any-return]


@overload
def same_size_output_padding(stride: int) -> int:
    ...


@overload
def same_size_output_padding(stride: SequenceType[int]) -> Tuple[int, ...]:
    ...


def same_size_output_padding(
    stride: Union[int, SequenceType[int]]
) -> Union[int, Tuple[int, ...]]:
    return elementwise(lambda x: x - 1, stride)  # type: ignore[no-any-return]


def is_valid_padding(padding: Union[int, SequenceType[int]]) -> bool:
    def is_valid(x: int) -> bool:
        return x > 0

    if isinstance(padding, int):
        return is_valid(padding)
    else:
        return all(elementwise(is_valid, padding))


def batch_up_image(
    image: torch.Tensor,
    desired_batch_size: Optional[int] = None,
    loader: Optional[DataLoader] = None,
) -> torch.Tensor:
    def extract_batch_size_from_loader(loader: DataLoader) -> int:
        batch_size = loader.batch_size
        if batch_size is not None:
            return batch_size

        try:  # type: ignore[unreachable]
            return loader.batch_sampler.batch_size
        except AttributeError:
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

    def try_seed_numpy(seed: int) -> None:
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

    def seed_torch(seed: int) -> None:
        torch.manual_seed(seed)

    def maybe_set_cudnn() -> None:
        if torch.backends.cudnn.is_available():
            # Both attributes are dynamically assigned to the module. See
            # https://github.com/pytorch/pytorch/blob/a1eaaea288cf51abcd69eb9b0993b1aa9c0ce41f/torch/backends/cudnn/__init__.py#L115-L129
            # The type errors are ignored, since this is still the recommended practice.
            # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    # the numpy random generator only accepts uint32 values
    seed = hash(seed) % (2 ** 32 - 1)

    maybe_seed_standard_library(seed)
    try_seed_numpy(seed)
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
    ext: str = ".pth",
    to_cpu: bool = True,
    hash_len: int = 8,
) -> str:
    if isinstance(input, nn.Module):
        state_dict = input.state_dict()
    else:
        state_dict = OrderedDict(input)

    if to_cpu:
        state_dict = OrderedDict(
            [(key, tensor.detach().cpu()) for key, tensor in state_dict.items()]
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
