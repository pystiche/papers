import itertools

import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche import gram_matrix, ops
from pystiche.image import extract_batch_size
from pystiche.loss import PerceptualLoss
from pystiche_papers.ulyanov_et_al_2016 import loss


def test_UlyanovEtAl2016FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            op = loss.UlyanovEtAl2016FeatureReconstructionOperator(
                encoder, impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_enc, target_enc)

            desired = score / extract_batch_size(input_enc) if impl_params else score

            assert actual == ptu.approx(desired)


def test_ulyanov_et_al_2016_content_loss(subtests):
    configs = (
        (True, True, 1e0),
        (True, False, 6e-1),
        (False, True, 1e0),
        (False, False, 1e0),
    )
    for impl_params, instance_norm, score_weight in configs:
        content_loss = loss.ulyanov_et_al_2016_content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            score_weight=score_weight,
        )
        assert isinstance(
            content_loss, loss.UlyanovEtAl2016FeatureReconstructionOperator
        )

        with subtests.test("layer"):
            assert content_loss.encoder.layer == "relu4_2"

        with subtests.test("score_weight"):
            assert content_loss.score_weight == pytest.approx(score_weight)


def test_UlyanovEtAl2016GramOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = gram_matrix(encoder(target_image), normalize=True)
    input_repr = gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, True, 1.0), (False, False, input_repr.size()[0]))
    for impl_params, normalize_by_num_channels, extra_batch_normalization in configs:
        with subtests.test(impl_params=impl_params):
            intern_target_repr = (
                target_repr / target_repr.size()[-1]
                if normalize_by_num_channels
                else target_repr
            )
            intern_inüut_repr = (
                input_repr / input_repr.size()[-1]
                if normalize_by_num_channels
                else input_repr
            )
            op = loss.UlyanovEtAl2016GramOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(intern_inüut_repr, intern_target_repr,)
            desired = score / extra_batch_normalization

            assert actual == ptu.approx(desired, rel=1e-3)


def test_ulyanov_et_al_2016_style_loss(subtests):
    for impl_params, instance_norm, stylization in itertools.product(
        (True, False), (True, False), (True, False)
    ):
        style_loss = loss.ulyanov_et_al_2016_style_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            stylization=stylization,
        )
        assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

        with subtests.test("encoding_ops"):
            assert all(
                isinstance(op, loss.UlyanovEtAl2016GramOperator)
                for op in style_loss.operators()
            )

        layers, layer_weights = zip(
            *[(op.encoder.layer, op.score_weight) for op in style_loss.operators()]
        )
        with subtests.test("layers"):
            desired_layers = (
                {"relu1_1", "relu2_1", "relu3_1", "relu4_1"}
                if impl_params and instance_norm
                else {"relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"}
            )
            assert set(layers) == desired_layers

        with subtests.test("layer_weights"):
            desired_layer_weights = (1e0,) * len(desired_layers)
            assert layer_weights == pytest.approx(desired_layer_weights)

        with subtests.test("score_weight"):
            if impl_params:
                if instance_norm:
                    score_weight = 1e0
                else:
                    score_weight = 1e3 if stylization else 1e0
            else:
                score_weight = 1e0
            assert style_loss.score_weight == pytest.approx(score_weight)


def test_ulyanov_et_al_2016_perceptual_loss(subtests):
    for stylization in (True, False):
        perceptual_loss = loss.ulyanov_et_al_2016_perceptual_loss(
            stylization=stylization
        )
        assert isinstance(perceptual_loss, PerceptualLoss)

        with subtests.test("content_loss"):
            assert (
                isinstance(
                    perceptual_loss.content_loss,
                    loss.UlyanovEtAl2016FeatureReconstructionOperator,
                )
                if stylization
                else perceptual_loss.content_loss is None
            )

        with subtests.test("style_loss"):
            assert isinstance(
                perceptual_loss.style_loss, loss.MultiLayerEncodingOperator
            )
