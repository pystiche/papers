import pytest


@pytest.fixture(scope="package", autouse=True)
def load_weights_mock(package_mocker):
    return package_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )
