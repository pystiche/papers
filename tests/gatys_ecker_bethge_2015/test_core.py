import pytest

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_ecker_bethge_2015 as paper
from pystiche.loss import PerceptualLoss


def test_gatys_ecker_bethge_2015_nst_smoke(
    subtests, mocker, content_image, style_image
):
    mock = mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015.core.default_image_optim_loop"
    )

    paper.gatys_ecker_bethge_2015_nst(content_image, style_image)
    input_image, criterion = mock.call_args[0]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, content_image)

    with subtests.test("criterion"):
        assert isinstance(criterion, PerceptualLoss)


def assert_is_rand_uniform(tensor, min=0.0, max=1.0):
    # FIXME: calculate abs with respect to confidence intervals
    actual_mean = torch.mean(tensor)
    desired_mean = ptu.approx((min + max) / 2, rel=0, abs=5e-2)
    assert (
        actual_mean == desired_mean
    ), f"mean mismatch: {actual_mean} != {desired_mean}"

    actual_var = torch.var(tensor)
    desired_var = ptu.approx((max - min) / 12, rel=0, abs=5e-2)
    assert actual_var == desired_var, f"var mismatch: {actual_var} != {desired_var}"


@pytest.mark.flaky
def test_gatys_ecker_bethge_2015_nst_smoke_not_impl_params(
    subtests, mocker, content_image, style_image
):
    mock = mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015.core.default_image_optim_loop"
    )
    paper.gatys_ecker_bethge_2015_nst(content_image, style_image, impl_params=False)
    input_image, _ = mock.call_args[0]

    assert_is_rand_uniform(input_image)
