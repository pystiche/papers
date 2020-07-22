import itertools

import numpy as np
import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche import extract_patches2d, ops
from pystiche.loss import PerceptualLoss
from pystiche.misc import to_2d_arg
from pystiche.ops.functional import mrf_loss, total_variation_loss
from pystiche_papers.li_wand_2016 import loss


def test_LiWand2016FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean"), (False, "sum"))
    for impl_params, loss_reduction in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.LiWand2016FeatureReconstructionOperator(
                encoder, impl_params=impl_params,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            desired = mse_loss(input_enc, target_enc, reduction=loss_reduction)

            assert actual == ptu.approx(desired)


def test_li_wand_2016_content_loss(subtests):
    content_loss = loss.li_wand_2016_content_loss()
    assert isinstance(content_loss, loss.LiWand2016FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    configs = ((True, 2e1), (False, 1e0))
    for impl_params, weight in configs:
        with subtests.test("score_weight"):
            content_loss = loss.li_wand_2016_content_loss(impl_params=impl_params)
            assert content_loss.score_weight == pytest.approx(weight)


def test_LiWand2016MRFOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, 3, 1, 1.0 / 2.0, "sum"), (False, 3, 1, 1.0, "sum"))
    for (
        impl_params,
        patch_size,
        stride,
        score_correction_factor,
        loss_reduction,
    ) in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.LiWand2016MRFOperator(encoder, 3, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

           extract_patches = loss.extract_normalized_patches2d if impl_params else extract_patches2d
           target_repr = extract_patches(target_enc, patch_size, stride)
           input_repr = extract_patches(input_enc, patch_size, stride)

            score = mrf_loss(input_repr, target_repr, reduction=loss_reduction)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_li_wand_2016_style_loss(subtests):
    style_loss = loss.li_wand_2016_style_loss()
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("encoding_ops"):
        assert all(isinstance(op, ops.MRFOperator) for op in style_loss.operators())

    configs = ((True, 1e-4, 2, 1, 5e-2, 1, 7.5), (False, 1e0, 1, 3, 5e-2, 2, 7.5))
    for (
        impl_params,
        score_weight,
        stride,
        num_scale_steps,
        scale_step_width,
        num_rotate_steps,
        rotate_step_width,
    ) in configs:
        style_loss = loss.li_wand_2016_style_loss(impl_params=impl_params)
        layers, layer_weights, op_stride, op_target_transforms = zip(
            *[
                (op.encoder.layer, op.score_weight, op.stride, op.target_transforms)
                for op in style_loss.operators()
            ]
        )

        with subtests.test("layers"):
            assert set(layers) == {"relu3_1", "relu4_1"}

        with subtests.test("stride"):
            assert op_stride == (to_2d_arg(stride),) * len(layers)

        with subtests.test("score_weight"):
            assert style_loss.score_weight == pytest.approx(score_weight)

        with subtests.test("layer_weights"):
            assert layer_weights == (1.0,) * len(layers)

        scaling_factors = np.arange(
            -num_scale_steps, num_scale_steps + 1, dtype=np.float
        )
        scaling_factors = 1.0 + (scaling_factors * scale_step_width)

        rotation_angles = np.arange(
            -num_rotate_steps, num_rotate_steps + 1, dtype=np.float
        )
        rotation_angles *= rotate_step_width

        for i, values in enumerate(itertools.product(scaling_factors, rotation_angles)):
            scaling_factor, rotation_angle = values
            with subtests.test("scaling_factor"):
                assert op_target_transforms[0][i].scaling_factor == scaling_factor

            with subtests.test("rotation_angles"):
                assert op_target_transforms[0][i].rotation_angle == rotation_angle


def test_LiWand2016TotalVariationOperator(subtests, input_image):
    configs = ((True, "sum", 1.0 / 2.0), (False, "sum", 1.0))
    for impl_params, loss_reduction, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.LiWand2016TotalVariationOperator(impl_params=impl_params,)
            actual = op(input_image)

            score = total_variation_loss(
                input_image, exponent=2.0, reduction=loss_reduction
            )

            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


def test_li_wand_2016_regularization(subtests):
    regularization_loss = loss.li_wand_2016_regularization()
    assert isinstance(regularization_loss, loss.LiWand2016TotalVariationOperator)

    with subtests.test("score_weight"):
        assert regularization_loss.score_weight == pytest.approx(1e-3)

    with subtests.test("exponent"):
        assert regularization_loss.exponent == pytest.approx(2.0)


def test_li_wand_2016_perceptual_loss(subtests):
    perceptual_loss = loss.li_wand_2016_perceptual_loss()
    assert isinstance(perceptual_loss, PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, loss.LiWand2016FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, loss.MultiLayerEncodingOperator)

    with subtests.test("regularization"):
        assert isinstance(
            perceptual_loss.regularization, loss.LiWand2016TotalVariationOperator
        )
