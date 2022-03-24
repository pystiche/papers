import pytest

from tests import mocks

import pystiche_papers.gatys_ecker_bethge_2016 as paper


@pytest.fixture(scope="package", autouse=True)
def multi_layer_encoder(package_mocker):
    setups = [((), {})]
    setups.extend(
        [((), dict(impl_params=impl_params)) for impl_params in (True, False)]
    )
    return mocks.patch_multi_layer_encoder_loader(
        targets=[
            mocks.make_mock_target("gatys_ecker_bethge_2016", *path)
            for path in (
                ("_loss", "_multi_layer_encoder"),
                ("_utils", "multi_layer_encoder_"),
            )
        ],
        loader=paper.multi_layer_encoder,
        setups=setups,
        mocker=package_mocker,
    )
