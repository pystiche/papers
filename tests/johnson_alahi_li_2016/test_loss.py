import itertools

import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche import gram_matrix, ops
from pystiche.loss import PerceptualLoss
from pystiche.ops.functional import total_variation_loss
from pystiche_papers.johnson_alahi_li_2016 import loss


@pytest.fixture
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


def test_get_content_score_weight_smoke(subtests, styles):
    for instance_norm, style in itertools.product((True, False), styles):
        with subtests.test(instance_norm=instance_norm, style=style):
            assert isinstance(
                loss.get_content_score_weight(instance_norm, style=style), float
            )


def test_johnson_alahi_li_2016_content_loss(subtests):
    content_loss = loss.johnson_alahi_li_2016_content_loss()
    assert isinstance(content_loss, ops.FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu2_2"

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(1e0)


def test_JohnsonAlahiLi2016GramOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = gram_matrix(encoder(target_image), normalize=True)
    input_repr = gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, target_repr.size()[1]), (False, 1.0))
    for impl_params, extra_num_channels_normalization in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.JohnsonAlahiLi2016GramOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score / extra_num_channels_normalization ** 2

            assert actual == ptu.approx(desired)


def test_get_style_score_weight_smoke(subtests, styles):
    for impl_params, instance_norm, style in itertools.product(
        (True, False), (True, False), styles
    ):
        with subtests.test(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        ):
            assert isinstance(
                loss.get_style_score_weight(impl_params, instance_norm, style=style),
                float,
            )


def test_johnson_alahi_li_2016_style_loss(subtests):
    style_loss = loss.johnson_alahi_li_2016_style_loss()
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("encoding_ops"):
        assert all(isinstance(op, ops.GramOperator) for op in style_loss.operators())

    layers, layer_weights = zip(
        *[(op.encoder.layer, op.score_weight) for op in style_loss.operators()]
    )
    with subtests.test("layers"):
        assert set(layers) == {"relu1_2", "relu2_2", "relu3_3", "relu4_3"}

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx([1.0] * len(layers))

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(5e0)


def test_JohnsonAlahiLi2016TotalVariationOperator(subtests, input_image):
    exponent = 2.0
    op = loss.JohnsonAlahiLi2016TotalVariationOperator(exponent=exponent)
    actual = op(input_image)

    desired = total_variation_loss(input_image, exponent=exponent, reduction="sum")

    assert actual == ptu.approx(desired)


def test_get_regularization_score_weight_smoke(subtests, styles):
    for instance_norm, style in itertools.product((True, False), styles):
        with subtests.test(instance_norm=instance_norm, style=style):
            assert isinstance(
                loss.get_regularization_score_weight(instance_norm, style=style), float,
            )


def test_johnson_alahi_li_2016_regularization(subtests):
    regularization = loss.johnson_alahi_li_2016_regularization()
    assert isinstance(regularization, loss.JohnsonAlahiLi2016TotalVariationOperator)

    with subtests.test("score_weight"):
        assert regularization.score_weight == pytest.approx(1e-6)


def test_johnson_alahi_li_2016_perceptual_loss(subtests):
    perceptual_loss = loss.johnson_alahi_li_2016_perceptual_loss()
    assert isinstance(perceptual_loss, PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, ops.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("regularization"):
        assert isinstance(
            perceptual_loss.regularization,
            loss.JohnsonAlahiLi2016TotalVariationOperator,
        )
