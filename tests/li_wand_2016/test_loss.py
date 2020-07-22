import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche_papers.li_wand_2016 import loss


def test_LiWand2016FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean"), (False, "sum"))
    for impl_params, loss_reduction in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.LiWand2016FeatureReconstructionOperator(
                encoder, impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            desired = mse_loss(input_enc, target_enc, reduction=loss_reduction)

            assert actual == ptu.approx(desired)


def test_li_wand_2016_content_loss(subtests):
    content_loss = loss.li_wand_2016_content_loss()
    assert isinstance(
        content_loss, loss.LiWand2016FeatureReconstructionOperator
    )

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    configs = ((True, 2e1), (False, 1e0))
    for impl_params, weight in configs:
        with subtests.test("score_weight"):
            content_loss = loss.li_wand_2016_content_loss(impl_params=impl_params)
            assert content_loss.score_weight == pytest.approx(weight)
