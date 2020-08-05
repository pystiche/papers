import contextlib
import importlib.util
import os
from os import path

from tests.utils import rmtree

__all__ = ["load_module", "dir_manager"]

REPLICATION_ROOT = path.abspath(
    path.join(path.dirname(__file__), "..", "..", "replication")
)


def load_module(rel_path, root=REPLICATION_ROOT):
    abs_path = path.join(root, rel_path)
    name, ext = path.splitext(path.basename(abs_path))
    if ext != ".py":
        abs_path = path.join(abs_path, "__init__.py")

    spec = importlib.util.spec_from_file_location(name, location=abs_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def dir_manager(rel_path, root=REPLICATION_ROOT):
    abs_path = path.join(root, rel_path)
    preexisting = _find_dirs(abs_path)
    try:
        yield
    finally:
        new = _find_dirs(abs_path) - preexisting
        for dir in new:
            rmtree(dir)


def _find_dirs(top):
    return {dir for dir, _, _ in os.walk(top)}
