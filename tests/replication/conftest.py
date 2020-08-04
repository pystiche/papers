import importlib.util
import os
import shutil
from os import path

import pytest


def load_module(location):
    name, ext = path.splitext(path.basename(location))
    if ext != ".py":
        location = path.join(location, "__init__.py")

    spec = importlib.util.spec_from_file_location(name, location=location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def rmtree(dir):
    if path.exists(dir):
        shutil.rmtree(dir, onerror=onerror)


def find_dirs(top):
    return {dir for dir, _, _ in os.walk(top)}


@pytest.fixture(scope="package")
def replication_root():
    here = path.dirname(__file__)
    return path.abspath(path.join(here, "..", "..", "replication"))


@pytest.fixture(scope="package")
def module_loader(replication_root):
    def module_loader_(rel_file):
        return load_module(path.join(replication_root, rel_file))

    return module_loader_


@pytest.fixture(scope="package")
def make_replication_dir_manager(replication_root):
    def make_replication_dir_manager_(paper):
        replication_dir = path.join(replication_root, paper)
        preexisting_dirs = find_dirs(replication_dir)
        yield
        new_dirs = find_dirs(replication_dir) - preexisting_dirs
        for dir in new_dirs:
            rmtree(dir)

    return make_replication_dir_manager_
