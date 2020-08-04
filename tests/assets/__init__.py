import functools
from os import path

from pystiche.image import read_image as _read_image

HERE = path.abspath(path.dirname(__file__))

__all__ = ["root", "read_image"]


def root():
    return path.dirname(__file__)


_read_image = functools.lru_cache()(_read_image)


def read_image(name):
    image = _read_image(path.join(HERE, "images", f"{name}.png"))
    # Since image is mutable we only cache the raw input and clone it for every call
    return image.clone()
