import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

import pystiche
import pystiche_papers.gatys_et_al_2017 as paper
from pystiche import loss, ops


def test_content_loss(subtests):
    content_loss = paper.content_loss()
    assert isinstance(content_loss, pystiche.loss.FeatureReconstructionLoss)

    hyper_parameters = paper.hyper_parameters().content_loss

    with subtests.test("layer"):
        assert content_loss.encoder.layer == hyper_parameters.layer

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_MultiLayerEncodingLoss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):

    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = pystiche.gram_matrix(encoder(target_image), normalize=True)
    input_repr = pystiche.gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, 1.0), (False, 1.0 / 4.0))
    for impl_params, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            loss = paper.MultiLayerEncodingLoss(
                multi_layer_encoder,
                (layer,),
                lambda encoder, layer_weight: pystiche.loss.GramLoss(
                    encoder, score_weight=layer_weight
                ),
                impl_params=impl_params,
                layer_weights="sum",
            )

            loss.set_target_image(target_image)
            actual = loss(input_image)

            score = mse_loss(input_repr, target_repr,)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_style_loss(subtests):
    style_loss = paper.style_loss()
    assert isinstance(style_loss, pystiche.loss.MultiLayerEncodingLoss)

    hyper_parameters = paper.hyper_parameters().style_loss

    with subtests.test("encoding_ops"):
        assert all(
            isinstance(loss, pystiche.loss.GramLoss) for loss in style_loss.Losss()
        )

    layers, layer_weights = zip(
        *[(loss.encoder.layer, loss.score_weight) for loss in style_loss.Losss()]
    )
    with subtests.test("layers"):
        assert layers == hyper_parameters.layers

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx(hyper_parameters.layer_weights)

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_guided_style_loss(subtests, content_guides):
    style_loss = paper.guided_style_loss(content_guides.keys())
    assert isinstance(style_loss, pystiche.loss.MultiRegionLoss)

    with subtests.test("encoding_losses"):
        assert all(
            isinstance(loss, paper.MultiLayerEncodingLoss)
            for loss in style_loss.Losss()
        )

    regions, region_weights = zip(
        *[(name, loss.score_weight) for name, loss in style_loss.named_Losss()]
    )

    with subtests.test("regions"):
        assert regions == tuple(content_guides.keys())

    with subtests.test("region_weights"):
        desired = (1e0,) * len(regions)
        assert region_weights == pytest.approx(desired)


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, pystiche.loss.FeatureReconstructionLoss,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, paper.MultiLayerEncodingLoss)


def test_guided_perceptual_loss(subtests, content_guides):

    perceptual_loss = paper.guided_perceptual_loss(content_guides.keys())
    assert isinstance(perceptual_loss, loss.GuidedPerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, pystiche.loss.FeatureReconstructionLoss,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, pystiche.loss.MultiRegionLoss)
