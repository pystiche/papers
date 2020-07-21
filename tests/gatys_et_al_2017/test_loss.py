import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche import gram_matrix, ops
from pystiche.loss import GuidedPerceptualLoss, PerceptualLoss
from pystiche.ops import FeatureReconstructionOperator
from pystiche_papers.gatys_et_al_2017 import loss


def test_gatys_et_al_2017_content_loss(subtests):
    content_loss = loss.gatys_et_al_2017_content_loss()
    assert isinstance(content_loss, FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(1e0)


def test_GatysEtAl2017StyleLoss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):

    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = gram_matrix(encoder(target_image), normalize=True)
    input_repr = gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, 1.0), (False, 1.0 / 4.0))
    for impl_params, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.GatysEtAl2017StyleLoss(
                multi_layer_encoder,
                (layer,),
                lambda encoder, layer_weight: ops.GramOperator(
                    encoder, score_weight=layer_weight
                ),
                impl_params=impl_params,
                layer_weights="sum",
            )

            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_gatys_et_al_2017_style_loss(subtests):

    style_loss = loss.gatys_et_al_2017_style_loss()
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("encoding_ops"):
        assert all(isinstance(op, ops.GramOperator) for op in style_loss.operators())

    layers, layer_weights = zip(
        *[(op.encoder.layer, op.score_weight) for op in style_loss.operators()]
    )
    with subtests.test("layers"):
        assert set(layers) == {"relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"}

    with subtests.test("layer_weights"):
        layers, nums_channels = zip(
            ("relu1_1", 64),
            ("relu2_1", 128),
            ("relu3_1", 256),
            ("relu4_1", 512),
            ("relu5_1", 512),
        )
        desired = tuple(1.0 / num_channels ** 2.0 for num_channels in nums_channels)
        assert layer_weights == pytest.approx(desired)

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(1e3)


def gatys_et_al_2017_guided_style_loss(subtests, guides):

    style_loss = loss.gatys_et_al_2017_guided_style_loss(guides.keys())
    assert isinstance(style_loss, ops.MultiRegionOperator)

    with subtests.test("encoding_ops"):
        assert all(
            isinstance(op, loss.GatysEtAl2017StyleLoss) for op in style_loss.operators()
        )

    regions, region_weights = zip(
        *[(op.encoder.layer, op.score_weight) for op in style_loss.operators()]
    )

    with subtests.test("regions"):
        assert regions == guides.keys()

    with subtests.test("region_weights"):
        desired = tuple(1.0) * len(regions)
        assert region_weights == pytest.approx(desired)


def test_gatys_et_al_2017_perceptual_loss(subtests):
    perceptual_loss = loss.gatys_et_al_2017_perceptual_loss()
    assert isinstance(perceptual_loss, PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, loss.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, loss.GatysEtAl2017StyleLoss)


def test_gatys_et_al_2017_guided_perceptual_loss(subtests, guides):

    perceptual_loss = loss.gatys_et_al_2017_guided_perceptual_loss(guides.keys())
    assert isinstance(perceptual_loss, GuidedPerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, loss.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, loss.MultiRegionOperator)
