import pytest

from pystiche_papers.li_wand_2016.utils import li_wand_2016_multi_layer_encoder


@pytest.fixture(scope="package")
def vgg_load_weights_mock(package_mocker):
    return package_mocker.patch(
        "pystiche.enc.models.vgg.VGGMultiLayerEncoder._load_weights"
    )


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder_mock(package_mocker, vgg_load_weights_mock):
    multi_layer_encoder = li_wand_2016_multi_layer_encoder()

    def new(impl_params=None):
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return package_mocker.patch(
        "pystiche_papers.li_wand_2016.loss.li_wand_2016_multi_layer_encoder", new,
    )
