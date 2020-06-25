import functools
from os import path

from pystiche.image import read_image as _read_image

HERE = path.abspath(path.dirname(__file__))

__all__ = ["read_image"]


@functools.lru_cache(maxsize=16)
def read_image(name):
    return _read_image(path.join(HERE, "images", f"{name}.png"))
