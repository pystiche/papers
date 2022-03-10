import pytest

import pytorch_testing_utils as ptu
import torch
from torch.nn.functional import mse_loss

import pystiche
import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import loss

from .utils import impl_params_and_instance_norm


@pytest.fixture(autouse=True)
def disable_autograd():
    with torch.autograd.no_grad():
        yield


@impl_params_and_instance_norm
def test_content_loss(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).content_loss

    content_loss = paper.content_loss(
        impl_params=impl_params, instance_norm=instance_norm,
    )
    assert isinstance(content_loss, pystiche.loss.FeatureReconstructionLoss)

    with subtests.test("layer"):
        assert content_loss.encoder.layer == hyper_parameters.layer

    with subtests.test("score_weight"):
        assert content_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_GramLoss(subtests, multi_layer_encoder_with_layer, target_image, input_image):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)

    configs = ((True, True), (False, False))
    for (impl_params, normalize_by_num_channels) in configs:
        with subtests.test(impl_params=impl_params):
            target_repr = pystiche.gram_matrix(encoder(target_image), normalize=True)
            input_repr = pystiche.gram_matrix(encoder(input_image), normalize=True)
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
            loss = paper.GramLoss(encoder, impl_params=impl_params)
            loss.set_target_image(target_image)
            actual = loss(input_image)

            desired = mse_loss(intern_input_repr, intern_target_repr)

            assert actual == ptu.approx(desired, rel=1e-3)


@impl_params_and_instance_norm
def test_style_loss(subtests, impl_params, instance_norm):
    hyper_parameters = paper.hyper_parameters(
        impl_params=impl_params, instance_norm=instance_norm
    ).style_loss

    style_loss = paper.style_loss(impl_params=impl_params, instance_norm=instance_norm,)
    assert isinstance(style_loss, pystiche.loss.MultiLayerEncodingLoss)

    with subtests.test("encoding_ops"):
        assert all(isinstance(loss, paper.GramLoss) for loss in style_loss.children())

    layers, layer_weights = zip(
        *[(loss.encoder.layer, loss.score_weight) for loss in style_loss.children()]
    )
    with subtests.test("layers"):
        assert layers == hyper_parameters.layers

    with subtests.test("layer_weights"):
        assert layer_weights == pytest.approx(hyper_parameters.layer_weights)

    with subtests.test("score_weight"):
        assert style_loss.score_weight == pytest.approx(hyper_parameters.score_weight)


def test_perceptual_loss(subtests):
    perceptual_loss = paper.perceptual_loss()
    assert isinstance(perceptual_loss, loss.PerceptualLoss)

    with subtests.test("content_loss"):
        assert isinstance(
            perceptual_loss.content_loss, pystiche.loss.FeatureReconstructionLoss,
        )

    with subtests.test("style_loss"):
        assert isinstance(
            perceptual_loss.style_loss, pystiche.loss.MultiLayerEncodingLoss
        )
