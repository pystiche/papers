import pytest

import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

import pystiche
import pystiche.ops.functional as F
import pystiche_papers.li_wand_2016 as paper
from pystiche import loss, misc, ops

from tests import utils


def test_FeatureReconstructionOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean"), (False, "sum"))
    for impl_params, loss_reduction in configs:
        with subtests.test(impl_params=impl_params):
            op = paper.FeatureReconstructionOperator(encoder, impl_params=impl_params,)
            op.set_target_image(target_image)
            actual = op(input_image)

            desired = mse_loss(input_enc, target_enc, reduction=loss_reduction)

            assert actual == ptu.approx(desired)


def test_content_loss(subtests):
    content_loss = paper.content_loss()
    assert isinstance(content_loss, paper.FeatureReconstructionOperator)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == "relu4_2"

    configs = ((True, 2e1), (False, 1e0))
    for impl_params, weight in configs:
        with subtests.test("score_weight"):
            content_loss = paper.content_loss(impl_params=impl_params)
            assert content_loss.score_weight == pytest.approx(weight)


def test_MRFOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    patch_size = 3
    stride = 1
    for impl_params in (True, False):
        with subtests.test(impl_params=impl_params):

            op = paper.MRFOperator(
                encoder, patch_size, impl_params=impl_params, stride=stride
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            extract_patches2d = (
                paper.extract_normalized_patches2d
                if impl_params
                else pystiche.extract_patches2d
            )
            target_repr = extract_patches2d(target_enc, patch_size, stride)
            input_repr = extract_patches2d(input_enc, patch_size, stride)

            desired = F.mrf_loss(input_repr, target_repr, reduction="sum")

            assert actual == ptu.approx(desired)


def test_style_loss(subtests):
    style_loss = paper.style_loss()
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("encoding_ops"):
        assert all(isinstance(op, ops.MRFOperator) for op in style_loss.operators())

    configs = ((True, 1e-4, 2), (False, 1e0, 1))
    for impl_params, score_weight, stride in configs:
        style_loss = paper.style_loss(impl_params=impl_params)
        layers, layer_weights, op_stride, op_target_transforms = zip(
            *[
                (op.encoder.layer, op.score_weight, op.stride, op.target_transforms)
                for op in style_loss.operators()
            ]
        )

        with subtests.test("layers"):
            assert set(layers) == {"relu3_1", "relu4_1"}

        with subtests.test("stride"):
            assert op_stride == (misc.to_2d_arg(stride),) * len(layers)

        with subtests.test("score_weight"):
            assert style_loss.score_weight == pytest.approx(score_weight)

        with subtests.test("layer_weights"):
            assert layer_weights == (1.0,) * len(layers)


@utils.parametrize_data(
    ("impl_params", "num_scale_steps", "num_rotate_steps"),
    pytest.param(True, 0, 0),
    pytest.param(False, 3, 2),
)
def test_style_loss_target_transforms(
    mocker, impl_params, num_scale_steps, num_rotate_steps
):
    mock = mocker.patch("pystiche_papers.li_wand_2016._loss._target_transforms")
    paper.style_loss(impl_params=impl_params)

    args = utils.call_args_to_namespace(mock.call_args, paper.target_transforms)

    assert args.impl_params is impl_params
    assert args.num_scale_steps == num_scale_steps
    assert args.scale_step_width == pytest.approx(5e-2)
    assert args.num_rotate_steps == num_rotate_steps
    assert args.rotate_step_width == pytest.approx(7.5)


def test_TotalVariationOperator(subtests, input_image):
    op = paper.TotalVariationOperator()
    actual = op(input_image)

    desired = F.total_variation_loss(input_image, exponent=op.exponent, reduction="sum")

    assert actual == ptu.approx(desired)


def test_regularization(subtests):
    regularization_loss = paper.regularization()
    assert isinstance(regularization_loss, paper.TotalVariationOperator)

    with subtests.test("score_weight"):
        assert regularization_loss.score_weight == pytest.approx(1e-3)

    with subtests.test("exponent"):
        assert regularization_loss.exponent == pytest.approx(2.0)


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, paper.FeatureReconstructionOperator,
        )

    with subtests.test("style_loss"):
        assert isinstance(perceptual_loss.style_loss, ops.MultiLayerEncodingOperator)

    with subtests.test("regularization"):
        assert isinstance(perceptual_loss.regularization, paper.TotalVariationOperator)
