import pytest

import pytorch_testing_utils as ptu
import torch
from torch import nn, optim

import pystiche
import pystiche_papers.li_wand_2016 as paper
from pystiche import enc
from pystiche.image import transforms

from tests import utils


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
    ("impl_params", "num_transforms"), pytest.param(True, 1), pytest.param(False, 35),
)
def test_target_transforms_smoke(impl_params, num_transforms):
    target_transforms = paper.target_transforms(impl_params=impl_params)
    assert len(target_transforms) == num_transforms


def test_target_transforms_call_smoke(target_image):
    for transform in paper.target_transforms(
        impl_params=True, num_scale_steps=1, num_rotate_steps=1
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
