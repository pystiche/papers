import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche
import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import enc, misc, ops

from tests.mocks import attach_method_mock


class TestOperator(paper.EncodingDiscriminatorOperator):
    def input_enc_to_repr(self, image):
        return image * 2.0

    def calculate_score(self, input_repr):
        self.accuracy = self.calculate_accuracy(input_repr)
        return torch.mean(input_repr)

    def calculate_accuracy(self, input_repr: torch.Tensor) -> torch.Tensor:
        comparator = torch.ge if self.real_images else torch.lt
        return torch.mean(comparator(input_repr, 0.0).float())


def test_EncodingDiscriminatorOperator_call(subtests, input_image):
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)

    for mode in (True, False):
        comparator = torch.ge if mode else torch.lt

        with subtests.test(mode=mode):
            test_op.real(mode)
            actual = test_op(input_image)
            prediction = encoder(input_image) * 2.0
            desired = torch.mean(prediction)
            ptu.assert_allclose(actual, desired)
            desired_accuracy = torch.mean(comparator(prediction, 0.0).float())
            ptu.assert_allclose(test_op.accuracy, desired_accuracy)


def test_EncodingDiscriminatorOperator_mode(subtests):
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)
    with subtests.test("real"):
        test_op.real()
        assert test_op.real_images
    with subtests.test("fake"):
        test_op.fake()
        assert not test_op.real_images


def test_PredictionOperator_call(subtests, input_image):
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    predictor = nn.Conv2d(3, 3, 1)

    op = paper.PredictionOperator(encoder, predictor)

    for mode in (True, False):
        comparator = torch.ge if mode else torch.lt

        with subtests.test(mode=mode):
            op.real(mode)
            actual = op(input_image)
            prediction = predictor(encoder(input_image))
            with subtests.test("loss"):
                desired = torch.mean(
                    binary_cross_entropy_with_logits(
                        prediction,
                        torch.ones_like(prediction)
                        if mode
                        else torch.zeros_like(prediction),
                    )
                )
                ptu.assert_allclose(actual, desired)
            with subtests.test("accuracy"):
                desired_accuracy = torch.mean(comparator(prediction, 0.0).float())
                ptu.assert_allclose(op.accuracy, desired_accuracy)


def test_MultiLayerPredictionOperator(subtests, input_image):
    def get_encoding_op(encoder, score_weight):
        return TestOperator(encoder, score_weight)

    layers = [str(index) for index in range(3)]
    modules = [(layer, nn.Conv2d(3, 3, 1)) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    multi_layer_prediction_op = paper.MultiLayerPredictionOperator(
        multi_layer_encoder, layers, get_encoding_op
    )

    for mode in (True, False):
        multi_layer_prediction_op.real(mode)
        multi_layer_prediction_op(input_image)
        with subtests.test(mode=mode):
            accuracies = [
                op.accuracy
                for op in multi_layer_prediction_op.discriminator_operators()
            ]
            desired = torch.mean(torch.stack(accuracies))
            ptu.assert_allclose(multi_layer_prediction_op.get_accuracy(), desired)


def test_prediction_loss(subtests):
    for impl_params in (True, False):
        prediction_loss = paper.prediction_loss(impl_params=impl_params,)
        assert isinstance(prediction_loss, paper.MultiLayerPredictionOperator)

        with subtests.test("encoding_ops"):
            assert all(
                isinstance(op, paper.PredictionOperator)
                for op in prediction_loss.discriminator_operators()
            )

        predictors, layers, layer_weights = zip(
            *[
                (op.predictor, op.encoder.layer, op.score_weight)
                for op in prediction_loss.discriminator_operators()
            ]
        )

        with subtests.test("predictor"):
            for predictor, kernel_size in zip(predictors, (5, 10, 10, 6, 3)):
                assert isinstance(predictor, nn.Module)
                assert predictor.kernel_size == misc.to_2d_arg(kernel_size)

        with subtests.test("layers"):
            desired_layers = {"0", "1", "3", "5", "6"}
            assert set(layers) == desired_layers

        with subtests.test("layer_weights"):
            desired_layer_weights = (1e0,) * len(desired_layers)
            assert layer_weights == pytest.approx(desired_layer_weights)

        with subtests.test("score_weight"):
            score_weight = 1e0 if impl_params else 1e-3
            assert prediction_loss.score_weight == pytest.approx(score_weight)


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
