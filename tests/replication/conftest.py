import sys

import pytest

from tests.mocks import make_mock_target


@pytest.fixture
def patch_argv(mocker):
    def patch_argv_(*args):
        return mocker.patch.object(sys, "argv", list(args))

    return patch_argv_


@pytest.fixture(scope="package", autouse=True)
def write_image(package_mocker):
    return package_mocker.patch(
        make_mock_target("image", "write_image", pkg="pystiche")
    )


@pytest.fixture(scope="package", autouse=True)
def save_state_dict(package_mocker):
    return package_mocker.patch(make_mock_target("utils", "save_state_dict"))
