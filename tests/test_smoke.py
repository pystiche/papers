import itertools
import os
import re
from importlib import import_module
from os import path
from setuptools import find_packages

import pytest

from ._utils import load_module

PROJECT_ROOT = path.abspath(path.join(path.dirname(__file__), ".."))
PACKAGE_NAME = "pystiche_papers"
PACKAGE_ROOT = path.join(PROJECT_ROOT, PACKAGE_NAME)


@pytest.fixture(scope="module")
def package_under_test():
    return load_module(PACKAGE_ROOT)


def test_import(subtests):
    def find_modules(dir, package=None):
        if package is not None:
            dir = path.join(dir, package.replace(".", os.sep))
        files = os.listdir(dir)
        modules = []
        for file in files:
            name, ext = path.splitext(file)
            if ext == ".py" and not name.startswith("_"):
                module = f"{package}." if package is not None else ""
                module += name
                modules.append(module)
        return modules

    public_packages = [
        package
        for package in find_packages(PACKAGE_ROOT)
        if not package.startswith("_")
    ]

    public_modules = find_modules(PACKAGE_ROOT)
    for package in public_packages:
        public_modules.extend(find_modules(PACKAGE_ROOT, package=package))

    for module in itertools.chain(public_packages, public_modules):
        with subtests.test(module=module):
            import_module(f".{module}", package=PACKAGE_NAME)


def test_about(subtests, package_under_test):
    for attr in (
        "name",
        "description",
        "base_version",
        "url",
        "license",
        "author",
        "author_email",
    ):
        with subtests.test(attr=attr):
            assert isinstance(getattr(package_under_test, f"__{attr}__"), str)

    attr = "is_dev_version"
    with subtests.test(attr=attr):
        assert isinstance(getattr(package_under_test, f"__{attr}__"), bool)


def test_name(package_under_test):
    assert package_under_test.__name__ == PACKAGE_NAME


def test_version(subtests, package_under_test):
    def is_canonical(version):
        # Copied from
        # https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )

    def is_dev(version):
        match = re.search(r"\+dev([.][\da-f]{7}([.]dirty)?)?$", version)
        if match is not None:
            return is_canonical(version[: match.span()[0]])
        else:
            return False

    with subtests.test():
        base_version = package_under_test.__base_version__
        assert is_canonical(base_version)

    with subtests.test():
        version = package_under_test.__version__
        assert is_canonical(version) or is_dev(version)
