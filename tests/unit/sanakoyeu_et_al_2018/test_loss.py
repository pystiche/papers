import pytest

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import ops


def test_transformed_image_loss(subtests):

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformed_image_loss = paper.transformed_image_loss(
                impl_params=impl_params
            )
            assert isinstance(transformed_image_loss, ops.FeatureReconstructionOperator)

            with subtests.test("score_weight"):
                assert (
                    transformed_image_loss.encoder.score_weight == pytest.approx(1e2)
                    if impl_params
                    else pytest.approx(1.0)
                )
