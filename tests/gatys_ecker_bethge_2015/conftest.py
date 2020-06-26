import pytest

from pystiche_papers.gatys_ecker_bethge_2015.utils import (
    gatys_ecker_bethge_2015_multi_layer_encoder,
)


@pytest.fixture(scope="package")
def vgg_load_weights_mock(package_mocker):
    return package_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder_mock(package_mocker, vgg_load_weights_mock):
    multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()

    def new(impl_params=None):
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return package_mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015.loss.gatys_ecker_bethge_2015_multi_layer_encoder",
        new,
    )
