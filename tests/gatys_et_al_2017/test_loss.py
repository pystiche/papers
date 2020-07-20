import pytest

from pystiche.ops import FeatureReconstructionOperator
from pystiche_papers.gatys_et_al_2017 import loss


def test_gatys_et_al_2017_content_loss(subtests):
    content_loss = loss.gatys_et_al_2017_content_loss()
    assert isinstance(content_loss, FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(1e0)
