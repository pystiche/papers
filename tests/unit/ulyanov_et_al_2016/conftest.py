import pytest

import pystiche_papers.ulyanov_et_al_2016 as paper

from tests import mocks


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder(package_mocker):
    return mocks.patch_multi_layer_encoder_loader(
        target=mocks.make_mock_target(
            "ulyanov_et_al_2016", "_loss", "_multi_layer_encoder"
        ),
        loader=paper.multi_layer_encoder,
        setup=((), {}),
        mocker=package_mocker,
    )
