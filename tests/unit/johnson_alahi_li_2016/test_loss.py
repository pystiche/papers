import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

import pystiche
import pystiche.pystiche.loss.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import loss, ops


def test_content_loss(subtests):
    content_loss = paper.content_loss()
    assert isinstance(content_loss, pystiche.loss.FeatureReconstructionLoss)

    hyper_parameters = paper.hyper_parameters().content_loss

    with subtests.test("layer"):
        assert content_loss.encoder.layer == hyper_parameters.layer


def test_GramLoss(subtests, multi_layer_encoder_with_layer, target_image, input_image):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = pystiche.gram_matrix(encoder(target_image), normalize=True)
    input_repr = pystiche.gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, target_repr.size()[1]), (False, 1.0))
    for impl_params, extra_num_channels_normalization in configs:
        with subtests.test(impl_params=impl_params):
            loss = paper.GramLoss(encoder, impl_params=impl_params,)
            loss.set_target_image(target_image)
            actual = loss(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score / extra_num_channels_normalization ** 2

            assert actual == ptu.approx(desired, rel=1e-3)


def test_style_loss(subtests):
    style_loss = paper.style_loss()
    assert isinstance(style_loss, pystiche.loss.MultiLayerEncodingLoss)

    hyper_parameters = paper.hyper_parameters().style_loss

    with subtests.test("encoding_ops"):
        assert all(isinstance(loss, pystiche.loss.GramLoss) for loss in style_loss.children())

    layers, layer_weights = zip(
        *[(loss.encoder.layer, loss.score_weight) for loss in style_loss.children()]
    )
    with subtests.test("layers"):
        assert layers == hyper_parameters.layers

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx((1.0,) * len(layers))

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_TotalVariationLoss(input_image):
    loss = paper.TotalVariationLoss()
    actual = loss(input_image)

    desired = F.total_variation_loss(input_image, reduction="sum")

    assert actual == ptu.approx(desired)


def test_regularization(subtests):
    regularization = paper.regularization()
    assert isinstance(regularization, paper.TotalVariationLoss)

    hyper_parameters = paper.hyper_parameters().regularization

    with subtests.test("score_weight"):
        assert regularization.score_weight == pytest.approx(
            hyper_parameters.score_weight
        )


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, pystiche.loss.FeatureReconstructionLoss,
        )

    with subtests.test("style_loss"):
        assert isinstance(
            perceptual_loss.style_loss, pystiche.loss.MultiLayerEncodingLoss
        )

    with subtests.test("regularization"):
        assert isinstance(perceptual_loss.regularization, paper.TotalVariationLoss,)
