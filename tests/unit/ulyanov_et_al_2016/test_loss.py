import itertools

import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

import pystiche
import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import image, loss, ops
from pystiche_papers import utils


def test_FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    batch_size = 4
    input_image = utils.batch_up_image(input_image, batch_size)
    target_image = utils.batch_up_image(target_image, batch_size)
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):
            op = paper.FeatureReconstructionOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_enc, target_enc)

            desired = (
                score / image.extract_batch_size(input_enc) if impl_params else score
            )
            assert actual == ptu.approx(desired)


def test_content_loss(subtests):
    configs = (
        (True, True, 1e0),
        (True, False, 6e-1),
        (False, True, 1e0),
        (False, False, 1e0),
    )
    for impl_params, instance_norm, score_weight in configs:
        content_loss = paper.content_loss(
            impl_params=impl_params,
            instance_norm=instance_norm,
            score_weight=score_weight,
        )
        assert isinstance(content_loss, paper.FeatureReconstructionOperator)

        with subtests.test("layer"):
            assert content_loss.encoder.layer == "relu4_2"

        with subtests.test("score_weight"):
            assert content_loss.score_weight == pytest.approx(score_weight)


def test_GramOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_repr = pystiche.gram_matrix(encoder(target_image), normalize=True)
    input_repr = pystiche.gram_matrix(encoder(input_image), normalize=True)

    configs = ((True, True, 1.0), (False, False, input_repr.size()[0]))
    for impl_params, normalize_by_num_channels, extra_batch_normalization in configs:
        with subtests.test(impl_params=impl_params):
            intern_target_repr = (
                target_repr / target_repr.size()[-1]
                if normalize_by_num_channels
                else target_repr
            )
            intern_input_repr = (
                input_repr / input_repr.size()[-1]
                if normalize_by_num_channels
                else input_repr
            )
            op = paper.GramOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(intern_input_repr, intern_target_repr,)
            desired = score / extra_batch_normalization

            assert actual == ptu.approx(desired, rel=1e-3)


def test_style_loss(subtests):
    for impl_params, instance_norm in itertools.product((True, False), (True, False)):
        style_loss = paper.style_loss(
            impl_params=impl_params, instance_norm=instance_norm,
        )
        assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

        with subtests.test("encoding_ops"):
            assert all(
                isinstance(op, paper.GramOperator) for op in style_loss.operators()
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
            score_weight = 1e3 if impl_params and not instance_norm else 1e0
            assert style_loss.score_weight == pytest.approx(score_weight)


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, paper.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, ops.MultiLayerEncodingOperator)
