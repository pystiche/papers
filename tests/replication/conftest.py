import pytest

from tests.mocks import make_mock_target


@pytest.fixture(scope="package", autouse=True)
def write_image(package_mocker):
    return package_mocker.patch(
        make_mock_target("image", "write_image", pkg="pystiche")
    )
