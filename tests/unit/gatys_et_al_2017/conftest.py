import pytest

import pystiche_papers.gatys_et_al_2017 as paper

from tests import mocks


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder(package_mocker):
    return mocks.patch_multi_layer_encoder_loader(
        targets=mocks.make_mock_target(
            "gatys_et_al_2017", "_loss", "_multi_layer_encoder"
        ),
        loader=paper.multi_layer_encoder,
        setups=((), {}),
        mocker=package_mocker,
    )
