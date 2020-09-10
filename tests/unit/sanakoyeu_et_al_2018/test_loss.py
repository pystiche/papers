

import pytest

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import ops


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


import pytest
import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import enc, misc


def test_EncodingDiscriminatorOperator_call(subtests, input_image):
    class TestOperator(paper.EncodingDiscriminatorOperator):
        def input_enc_to_repr(self, image):
            return image * 2.0

        def calculate_score(self, input_repr):
            self.accuracy = self.calculate_accuracy(input_repr)
            return torch.mean(input_repr)

        def calculate_accuracy(self, input_repr: torch.Tensor) -> torch.Tensor:
            comparator = torch.ge if self._target_distribution else torch.lt
            return torch.mean(comparator(input_repr, 0.0).float())

    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))

    test_op = TestOperator(encoder)

    for mode in (True, False):
        comparator = torch.ge if mode else torch.lt

        with subtests.test(mode=mode):
            test_op.real() if mode else test_op.fake()
            actual = test_op(input_image)
            prediction = encoder(input_image) * 2.0
            desired = torch.mean(prediction)
            ptu.assert_allclose(actual, desired)
            desired_accuracy = torch.mean(comparator(prediction, 0.0).float())
            ptu.assert_allclose(test_op.accuracy, desired_accuracy)


def test_PredictionOperator_call(subtests, input_image):
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    predictor = nn.Conv2d(3, 3, 1)

    op = paper.PredictionOperator(encoder, predictor)

    for mode in (True, False):
        comparator = torch.ge if mode else torch.lt

        with subtests.test(mode=mode):
            op.real() if mode else op.fake()
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
    class TestOperator(paper.EncodingDiscriminatorOperator):
        def input_enc_to_repr(self, image):
            return image * 2.0

        def calculate_score(self, input_repr):
            self.accuracy = self.calculate_accuracy(input_repr)
            return torch.mean(input_repr)

        def calculate_accuracy(self, input_repr: torch.Tensor) -> torch.Tensor:
            comparator = torch.ge if self._target_distribution else torch.lt
            return torch.mean(comparator(input_repr, 0.0).float())

    def get_encoding_op(encoder, score_weight):
        return TestOperator(encoder, score_weight)

    layers = [str(index) for index in range(3)]
    modules = [(layer, nn.Conv2d(3, 3, 1)) for layer in layers]
    multi_layer_encoder = enc.MultiLayerEncoder(modules)

    multi_layer_prediction_op = paper.MultiLayerPredictionOperator(
        multi_layer_encoder, layers, get_encoding_op
    )

    for mode in (True, False):
        multi_layer_prediction_op.real() if mode else multi_layer_prediction_op.fake()
        _ = multi_layer_prediction_op(input_image)
        with subtests.test(mode=mode):
            with subtests.test("operator mode"):
                for op in multi_layer_prediction_op.discriminator_operators():
                    assert op._target_distribution == mode

            with subtests.test("accuracy"):
                desired = torch.mean(
                    torch.stack(
                        [
                            op.accuracy
                            for op in multi_layer_prediction_op.discriminator_operators()
                        ]
                    )
                )
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


def test_discriminator_loss():
    discriminator_loss = paper.discriminator_loss()
    assert isinstance(discriminator_loss, paper.DiscriminatorLoss)
