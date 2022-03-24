import pytest

from tests import utils

import pytorch_testing_utils as ptu
import torch
from torch import nn, optim

import pystiche
import pystiche_papers.li_wand_2016 as paper
from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters


def test_hyper_parameters(subtests):
    hyper_parameters = paper.hyper_parameters()
    assert isinstance(hyper_parameters, HyperParameters)


@utils.impl_params
def test_hyper_parameters_content_loss(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "content_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layer == "relu4_1" if impl_params else "relu4_2"

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(
            2e1 if impl_params else 1e0
        )


@utils.impl_params
def test_hyper_parameters_target_transforms(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "target_transforms"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("num_scale_steps"):
        assert hyper_parameters.num_scale_steps == 0 if impl_params else 3

    with subtests.test("scale_step_width"):
        assert hyper_parameters.scale_step_width == pytest.approx(5e-2)

    with subtests.test("num_rotate_steps"):
        assert hyper_parameters.num_rotate_steps == 0 if impl_params else 2

    with subtests.test("rotate_step_width"):
        assert hyper_parameters.rotate_step_width == pytest.approx(7.5)


@utils.impl_params
def test_hyper_parameters_style_loss(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "style_loss"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("layer"):
        assert hyper_parameters.layers == ("relu3_1", "relu4_1")

    with subtests.test("layer_weights"):
        assert hyper_parameters.layer_weights == "sum"

    with subtests.test("patch_size"):
        assert hyper_parameters.patch_size == 3

    with subtests.test("stride"):
        assert hyper_parameters.stride == 2 if impl_params else 1

    with subtests.test("score_weight"):
        assert hyper_parameters.score_weight == pytest.approx(
            1e-4 if impl_params else 1e0
        )


@utils.impl_params
def test_hyper_parameters_image_pyramid(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "image_pyramid"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("max_edge_size"):
        assert hyper_parameters.max_edge_size == 384

    with subtests.test("num_steps"):
        assert hyper_parameters.num_steps == 100 if impl_params else 200

    with subtests.test("num_levels"):
        if impl_params:
            assert hyper_parameters.num_levels == 3
        else:
            assert hyper_parameters.num_levels is None

    with subtests.test("min_edge_size"):
        assert hyper_parameters.min_edge_size == 64

    with subtests.test("edge"):
        assert hyper_parameters.edge == "long"


@utils.impl_params
def test_hyper_parameters_nst(subtests, impl_params):
    hyper_parameters = paper.hyper_parameters(impl_params=impl_params)

    sub_params = "nst"
    assert sub_params in hyper_parameters
    hyper_parameters = getattr(hyper_parameters, sub_params)

    with subtests.test("starting_point"):
        assert hyper_parameters.starting_point == "content" if impl_params else "random"


def test_extract_normalized_patches2d(subtests):
    height = 4
    width = 4
    patch_size = 2
    stride = 1

    input = torch.ones(1, 1, height, width).requires_grad_(True)
    input_normalized = torch.ones(1, 1, height, width).requires_grad_(True)
    target = torch.zeros(1, 1, height, width).detach()

    input_patches = pystiche.extract_patches2d(
        input, patch_size=patch_size, stride=stride
    )
    input_patches_normalized = paper.extract_normalized_patches2d(
        input_normalized, patch_size=patch_size, stride=stride
    )
    target_patches = pystiche.extract_patches2d(
        target, patch_size=patch_size, stride=stride
    )

    loss = 0.5 * torch.sum((input_patches - target_patches) ** 2.0)
    loss.backward()

    loss_normalized = 0.5 * torch.sum(
        (input_patches_normalized - target_patches) ** 2.0
    )
    loss_normalized.backward()

    with subtests.test("forward"):
        ptu.assert_allclose(input_patches_normalized, input_patches)

    with subtests.test("backward"):
        ptu.assert_allclose(input_normalized.grad, torch.ones_like(input_normalized))


def test_extract_normalized_patches2d_no_overlap(subtests):
    height = 4
    width = 4
    patch_size = 2
    stride = 2

    input = torch.ones(1, 1, height, width).requires_grad_(True)
    input_normalized = torch.ones(1, 1, height, width).requires_grad_(True)
    target = torch.zeros(1, 1, height, width).detach()

    input_patches = pystiche.extract_patches2d(
        input, patch_size=patch_size, stride=stride
    )
    input_patches_normalized = paper.extract_normalized_patches2d(
        input_normalized, patch_size=patch_size, stride=stride
    )
    target_patches = pystiche.extract_patches2d(
        target, patch_size=patch_size, stride=stride
    )

    loss = 0.5 * torch.sum((input_patches - target_patches) ** 2.0)
    loss.backward()

    loss_normalized = 0.5 * torch.sum(
        (input_patches_normalized - target_patches) ** 2.0
    )
    loss_normalized.backward()

    with subtests.test("forward"):
        ptu.assert_allclose(input_patches_normalized, input_patches)

    with subtests.test("backward"):
        ptu.assert_allclose(input_normalized.grad, input.grad)


@utils.parametrize_data(
    ("impl_params", "num_transforms"),
    pytest.param(True, 1),
    pytest.param(False, 35),
)
def test_target_transforms_smoke(impl_params, num_transforms):
    target_transforms = paper.target_transforms(impl_params=impl_params)
    assert len(target_transforms) == num_transforms


def test_target_transforms_call_smoke(target_image):
    hyper_parameters = paper.hyper_parameters(impl_params=True)
    hyper_parameters.target_transforms.num_scale_steps = 1
    hyper_parameters.target_transforms.num_rotate_steps = 1
    for transform in paper.target_transforms(
        impl_params=True, hyper_parameters=hyper_parameters
    ):
        assert isinstance(transform(target_image), torch.Tensor)


def test_preprocessor():
    assert isinstance(paper.preprocessor(), transforms.CaffePreprocessing)


def test_postprocessor():
    assert isinstance(paper.postprocessor(), transforms.CaffePostprocessing)


@pytest.mark.slow
def test_multi_layer_encoder(subtests):
    multi_layer_encoder = paper.multi_layer_encoder()
    assert isinstance(multi_layer_encoder, enc.VGGMultiLayerEncoder)

    with subtests.test("internal preprocessing"):
        assert "preprocessing" not in multi_layer_encoder

    with subtests.test("inplace"):
        relu_modules = [
            module
            for module in multi_layer_encoder.modules()
            if isinstance(module, nn.ReLU)
        ]
        assert all(module.inplace for module in relu_modules)


def test_optimizer(subtests, input_image):
    params = input_image
    optimizer = paper.optimizer(params)

    assert isinstance(optimizer, optim.LBFGS)
    assert len(optimizer.param_groups) == 1

    param_group = optimizer.param_groups[0]

    with subtests.test(msg="optimization params"):
        assert len(param_group["params"]) == 1
        assert param_group["params"][0] is params

    with subtests.test(msg="optimizer properties"):
        assert param_group["lr"] == ptu.approx(1.0)
        assert param_group["max_iter"] == 1
