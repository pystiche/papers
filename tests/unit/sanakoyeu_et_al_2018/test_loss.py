
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

import pytorch_testing_utils as ptu
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche import enc


def test_DiscriminatorEncodingOperator_call(subtests):
    torch.manual_seed(0)
    image = torch.rand(1, 3, 128, 128)
    encoder = enc.SequentialEncoder((nn.Conv2d(3, 3, 1),))
    prediction_module = nn.Conv2d(3, 3, 1)

    op = paper.DiscriminatorEncodingOperator(encoder, prediction_module)

    for real in (True, False):
        with subtests.test(real=real):
            actual = op(image, real)
            prediction = prediction_module(encoder(image))

            with subtests.test("loss"):
                desired = binary_cross_entropy_with_logits(
                    prediction,
                    torch.ones_like(prediction)
                    if real
                    else torch.zeros_like(prediction),
                )
                ptu.assert_allclose(actual, desired)

            with subtests.test("accuracy"):
                actual = op.get_current_acc
                desired = (
                    torch.mean(
                        torch.masked_fill(
                            torch.zeros_like(prediction),
                            prediction > torch.zeros_like(prediction),
                            1,
                        )
                    )
                    if real
                    else torch.masked_fill(
                        torch.zeros_like(prediction),
                        prediction < torch.zeros_like(prediction),
                        1,
                    )
                )
                ptu.assert_allclose(actual, desired)
