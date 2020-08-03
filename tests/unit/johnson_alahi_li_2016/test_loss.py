import itertools

import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import loss, ops


def test_content_loss(subtests):
    content_loss = paper.content_loss()
    assert isinstance(content_loss, ops.FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu2_2"

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(1e0)


def test_content_loss_score_weight_smoke(subtests, styles):
    for instance_norm, style in itertools.product((True, False), styles):
        with subtests.test(instance_norm=instance_norm, style=style):
            content_loss = paper.content_loss(instance_norm=instance_norm, style=style)
            assert isinstance(content_loss.score_weight, float)


def test_GramOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = pystiche.gram_matrix(encoder(target_image), normalize=True)
    input_repr = pystiche.gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, target_repr.size()[1]), (False, 1.0))
    for impl_params, extra_num_channels_normalization in configs:
        with subtests.test(impl_params=impl_params):
            op = paper.GramOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score / extra_num_channels_normalization ** 2

            assert actual == ptu.approx(desired, rel=1e-3)


def test_style_loss(subtests):
    style_loss = paper.style_loss()
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


def test_style_loss_score_weight_smoke(subtests, styles):
    for impl_params, instance_norm, style in itertools.product(
        (True, False), (True, False), styles
    ):
        with subtests.test(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        ):
            style_loss = paper.style_loss(
                impl_params=impl_params, instance_norm=instance_norm, style=style
            )
            assert isinstance(style_loss.score_weight, float)


def test_TotalVariationOperator(subtests, input_image):
    exponent = 2.0
    op = paper.TotalVariationOperator(exponent=exponent)
    actual = op(input_image)

    desired = F.total_variation_loss(input_image, exponent=exponent, reduction="sum")

    assert actual == ptu.approx(desired)


def test_regularization(subtests):
    regularization = paper.regularization()
    assert isinstance(regularization, paper.TotalVariationOperator)

    with subtests.test("score_weight"):
        assert regularization.score_weight == pytest.approx(1e-6, rel=1e-3)


def test_regularization_score_weight_smoke(subtests, styles):
    for impl_params, instance_norm, style in itertools.product(
        (True, False), (True, False), styles
    ):
        with subtests.test(
            impl_params=impl_params, instance_norm=instance_norm, style=style
        ):
            style_loss = paper.style_loss(
                impl_params=impl_params, instance_norm=instance_norm, style=style
            )
            assert isinstance(style_loss.score_weight, float)


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, ops.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("regularization"):
        assert isinstance(perceptual_loss.regularization, paper.TotalVariationOperator,)
