import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.nn.functional import l1_loss

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import enc, loss, ops

from tests.mocks import attach_method_mock


@pytest.fixture
def prediction_loss_mocks(mocker):
    mock = mocker.Mock(
        side_effect=lambda image: pystiche.LossDict([("0", torch.mean(image))])
    )
    attach_method_mock(mock, "get_accuracy", return_value=torch.Tensor([0.5]))
    attach_method_mock(mock, "real", return_value=None)
    attach_method_mock(mock, "fake", return_value=None)
    patch = mocker.patch(
        "pystiche_papers.sanakoyeu_et_al_2018._loss.MultiLayerPredictionOperator",
        return_value=mock,
    )
    return patch, mock


def test_DiscriminatorLoss(
    subtests, prediction_loss_mocks, input_image, style_image, content_image
):
    patch, mock = prediction_loss_mocks
    discriminator_loss = paper.DiscriminatorLoss(mock)

    for input_photo in (None, content_image):
        with subtests.test(input_photo=input_photo):
            with subtests.test("loss"):
                actual = discriminator_loss(input_image, style_image, input_photo)
                desired = torch.mean(input_image) + torch.mean(style_image)
                if input_photo is not None:
                    desired += torch.mean(input_photo)
                ptu.assert_allclose(actual, desired)
            with subtests.test("accuracy"):
                ptu.assert_allclose(discriminator_loss.accuracy, mock.get_accuracy())


def test_discriminator_loss_smoke():
    discriminator_loss = paper.discriminator_loss()
    assert isinstance(discriminator_loss, paper.DiscriminatorLoss)


def test_transformed_image_loss(subtests):

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            transformed_image_loss = paper.transformed_image_loss(
                impl_params=impl_params
            )
            assert isinstance(transformed_image_loss, ops.FeatureReconstructionOperator)

            with subtests.test("score_weight"):
                assert (
                    transformed_image_loss.score_weight == pytest.approx(1e2)
                    if impl_params
                    else pytest.approx(1.0)
                )


def test_MAEReconstructionOperator_call():
    torch.manual_seed(0)
    target_image = torch.rand(1, 3, 128, 128)
    input_image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    op = paper.MAEFeatureReconstructionOperator(encoder)
    op.set_target_image(target_image)

    actual = op(input_image)
    desired = l1_loss(encoder(input_image), encoder(target_image))
    ptu.assert_allclose(actual, desired)


def test_style_aware_content_loss(subtests, multi_layer_encoder_with_layer):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    for impl_params, score_weight in ((True, 1e2), (False, 1e0)):
        with subtests.test(impl_params=impl_params):
            content_loss = paper.style_aware_content_loss(
                encoder, impl_params=impl_params
            )
            assert isinstance(
                content_loss,
                paper.MAEFeatureReconstructionOperator
                if impl_params
                else ops.FeatureReconstructionOperator,
            )
            with subtests.test("score_weight"):
                assert content_loss.score_weight == pytest.approx(score_weight)


def test_transformer_loss(subtests, multi_layer_encoder_with_layer):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    transformer_loss = paper.transformer_loss(encoder)
    assert isinstance(transformer_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(transformer_loss.content_loss, ops.OperatorContainer)
        with subtests.test("operator"):
            for module in transformer_loss.content_loss.children():
                assert isinstance(module, ops.EncodingComparisonOperator)

    with subtests.test("style_loss"):
        assert isinstance(
            transformer_loss.style_loss, paper.MultiLayerPredictionOperator
        )
