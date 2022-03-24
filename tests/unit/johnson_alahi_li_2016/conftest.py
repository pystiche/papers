import pytest

from tests import mocks

import pystiche_papers.johnson_alahi_li_2016 as paper


@pytest.fixture(scope="package")
def styles():
    return (
        "composition_vii",
        "feathers",
        "la_muse",
        "mosaic",
        "starry_night",
        "the_scream",
        "udnie",
        "the_wave",
    )


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder(package_mocker):
    return mocks.patch_multi_layer_encoder_loader(
        targets=mocks.make_mock_target(
            "johnson_alahi_li_2016", "_loss", "_multi_layer_encoder"
        ),
        loader=paper.multi_layer_encoder,
        setups=((), {"impl_params": True}),
        mocker=package_mocker,
    )
