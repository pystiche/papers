import pytest

from torch.nn.functional import mse_loss

import pytorch_testing_utils as ptu
from pystiche import gram_matrix, ops
from pystiche.loss import PerceptualLoss
from pystiche_papers.gatys_ecker_bethge_2015 import loss
from pystiche_papers.gatys_ecker_bethge_2015.utils import (
    gatys_ecker_bethge_2015_multi_layer_encoder,
)


@pytest.fixture(scope="module", autouse=True)
def multi_layer_encoder_mock(module_mocker, load_weights_mocks):
    multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()

    def new(impl_params=None):
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    return module_mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015.loss.gatys_ecker_bethge_2015_multi_layer_encoder",
        new,
    )


def test_GatysEckerBethge2015FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean", 1.0), (False, "sum", 1.0 / 2.0))
    for impl_params, loss_reduction, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.GatysEckerBethge2015FeatureReconstructionOperator(
                encoder, impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_enc, target_enc, reduction=loss_reduction)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_gatys_ecker_bethge_2015_content_loss(subtests):
    content_loss = loss.gatys_ecker_bethge_2015_content_loss()
    assert isinstance(
        content_loss, loss.GatysEckerBethge2015FeatureReconstructionOperator
    )

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(1e0)


def test_GatysEckerBethge2015StyleLoss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = gram_matrix(encoder(target_image), normalize=True)
    input_repr = gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, 1.0), (False, 1.0 / 4.0))
    for impl_params, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.GatysEckerBethge2015StyleLoss(
                multi_layer_encoder,
                (layer,),
                lambda encoder, layer_weight: ops.GramOperator(
                    encoder, score_weight=layer_weight
                ),
                impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_get_layer_weights():
    layers, nums_channels = zip(
        ("relu1_1", 64),
        ("relu2_1", 128),
        ("relu3_1", 256),
        ("relu4_1", 512),
        ("relu5_1", 512),
    )

    actual = loss.get_layer_weights(layers)
    desired = tuple(1.0 / num_channels ** 2.0 for num_channels in nums_channels)
    assert actual == pytest.approx(desired)


def test_gatys_ecker_bethge_2015_style_loss(subtests):
    style_loss = loss.gatys_ecker_bethge_2015_style_loss()
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("encoding_ops"):
        assert all(isinstance(op, ops.GramOperator) for op in style_loss.operators())

    layers, layer_weights = zip(
        *[(op.encoder.layer, op.score_weight) for op in style_loss.operators()]
    )
    with subtests.test("layers"):
        assert set(layers) == {"relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"}

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx(loss.get_layer_weights(layers))

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(1e3)


def test_gatys_ecker_bethge_2015_perceptual_loss(subtests):
    perceptual_loss = loss.gatys_ecker_bethge_2015_perceptual_loss()
    assert isinstance(perceptual_loss, PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss,
            loss.GatysEckerBethge2015FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(
            perceptual_loss.style_loss, loss.GatysEckerBethge2015StyleLoss
        )
