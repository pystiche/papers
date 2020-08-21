import pytest

import pystiche_papers.johnson_alahi_li_2016 as paper

from tests import mocks


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
        target=mocks.make_mock_target(
            "johnson_alahi_li_2016", "_loss", "_multi_layer_encoder"
        ),
        loader=paper.multi_layer_encoder,
        setup=((), {"impl_params": True}),
        mocker=package_mocker,
    )
