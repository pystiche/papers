import torch

import pystiche_papers.gatys_ecker_bethge_2015 as paper
import pytorch_testing_utils as ptu
from pystiche.loss import PerceptualLoss


def test_gatys_ecker_bethge_2015_nst(subtests, mocker, content_image, style_image):
    mock = mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015.core.default_image_optim_loop"
    )

    impl_params = True
    paper.gatys_ecker_bethge_2015_nst(
        content_image, style_image, impl_params=impl_params
    )
    input_image, criterion = mock.call_args[0]

    with subtests.test("input_image", impl_params=impl_params):
        ptu.assert_allclose(input_image, content_image)

    with subtests.test("criterion"):
        assert isinstance(criterion, PerceptualLoss)

    impl_params = False
    paper.gatys_ecker_bethge_2015_nst(
        content_image, style_image, impl_params=impl_params
    )
    input_image, _ = mock.call_args[0]

    with subtests.test("input_image", impl_params=impl_params):
        # mean and variance of a uniform distribution over interval [0, 1]
        assert torch.mean(input_image) == ptu.approx(1 / 2, rel=5e-2)
        assert torch.var(input_image) == ptu.approx(1 / 12, rel=5e-2)
