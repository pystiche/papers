
import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche.image import extract_batch_size
from pystiche_papers.ulyanov_et_al_2016 import loss


def test_UlyanovEtAl2016FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            op = loss.UlyanovEtAl2016FeatureReconstructionOperator(
                encoder, impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_enc, target_enc)

            desired = score / extract_batch_size(input_enc) if impl_params else score

            assert actual == ptu.approx(desired)


def test_ulyanov_et_al_2016_content_loss(subtests):
    configs = (
        (True, True, 1e0),
        (True, False, 6e-1),
        (False, True, 1e0),
        (False, False, 1e0),
    )
    for impl_params, instance_norm, score_weight in configs:
        content_loss = loss.ulyanov_et_al_2016_content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            score_weight=score_weight,
        )
        assert isinstance(
            content_loss, loss.UlyanovEtAl2016FeatureReconstructionOperator
        )

        with subtests.test("layer"):
            assert content_loss.encoder.layer == "relu4_2"

        with subtests.test("score_weight"):
            assert content_loss.score_weight == pytest.approx(score_weight)