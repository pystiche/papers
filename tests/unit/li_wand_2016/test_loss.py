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


@utils.impl_params
def test_content_loss(subtests, impl_params):
    content_loss = paper.content_loss(impl_params=impl_params)
    assert isinstance(content_loss, paper.FeatureReconstructionOperator)

    hyper_parameters = paper.hyper_parameters(impl_params=impl_params).content_loss

    with subtests.test("layer"):
        assert content_loss.encoder.layer == hyper_parameters.layer

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_MRFOperator(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    patch_size = 3
    stride = 1
    configs = ((True, 1.0 / 2.0), (False, 1.0))
    for (impl_params, score_correction_factor,) in configs:
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

            score = F.mrf_loss(input_repr, target_repr, reduction="sum")
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


@utils.impl_params
def test_style_loss(subtests, impl_params):
    style_loss = paper.style_loss(impl_params=impl_params)
    assert isinstance(style_loss, ops.MultiLayerEncodingOperator)

    hyper_parameters = paper.hyper_parameters(impl_params=impl_params).style_loss

    with subtests.test("ops"):
        assert all(isinstance(op, paper.MRFOperator) for op in style_loss.operators())

    layers, layer_weights, patch_size, stride = zip(
        *[
            (op.encoder.layer, op.score_weight, op.patch_size, op.stride)
            for op in style_loss.operators()
        ]
    )
    with subtests.test("layers"):
        assert layers == hyper_parameters.layers

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx((1.0,) * len(layers))

    with subtests.test("patch_size"):
        assert patch_size == (misc.to_2d_arg(hyper_parameters.patch_size),) * len(
            layers
        )

    with subtests.test("stride"):
        assert stride == (misc.to_2d_arg(hyper_parameters.stride),) * len(layers)

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


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
    configs = ((True, 1.0 / 2.0), (False, 1.0))
    for impl_params, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = paper.TotalVariationOperator(impl_params=impl_params,)
            actual = op(input_image)

            score = F.total_variation_loss(
                input_image, exponent=op.exponent, reduction="sum"
            )

            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)


@utils.impl_params
def test_regularization(subtests, impl_params):
    regularization_loss = paper.regularization(impl_params=impl_params)
    assert isinstance(regularization_loss, paper.TotalVariationOperator)

    hyper_parameters = paper.hyper_parameters(impl_params=impl_params).regularization

    with subtests.test("score_weight"):
        assert regularization_loss.score_weight == pytest.approx(
            hyper_parameters.score_weight
        )


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
