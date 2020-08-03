from typing import Callable

import numpy as np

import pytorch_testing_utils as ptu
import torch

import pystiche_papers.gatys_ecker_bethge_2015 as paper
from tests.utils import is_callable


def test_nst_smoke(subtests, mocker, content_image, style_image):
    mock = mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015._nst.optim.default_image_optim_loop"
    )

    paper.nst(content_image, style_image)

    args, kwargs = mock.call_args
    input_image, criterion = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, content_image)

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.perceptual_loss()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(optimizer, type(paper.optimizer(input_image)))

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(paper.preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(paper.postprocessor()))


# TODO: find a better place for this
def two_sample_kolmogorov_smirnov_test(
    samples: torch.Tensor,
    cdf_fn: Callable[[torch.Tensor], torch.Tensor],
    significance_level: float = 5e-2,
) -> bool:
    numel = samples.numel()
    ecdf = torch.arange(1, numel + 1, dtype=samples.dtype) / numel
    cdf = cdf_fn(torch.sort(samples.flatten())[0])
    statistic = torch.max(torch.abs(ecdf - cdf)).item()
    # See
    # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
    # for the special case n == m
    threshold = np.sqrt(-np.log(significance_level / 2) / numel)
    return statistic > threshold


# TODO: find a better place for this
def assert_is_rand_uniform(samples, min=0.0, max=1.0, significance_level=5e-2):
    assert torch.all(samples >= min)
    assert torch.all(samples <= max)

    def cdf_fn(x):
        return x

    assert not two_sample_kolmogorov_smirnov_test(
        samples, cdf_fn, significance_level=significance_level
    )


def test_nst_smoke_not_impl_params(subtests, mocker, content_image, style_image):
    mock = mocker.patch(
        "pystiche_papers.gatys_ecker_bethge_2015._nst.optim.default_image_optim_loop"
    )
    paper.nst(content_image, style_image, impl_params=False)
    input_image, _ = mock.call_args[0]

    assert_is_rand_uniform(input_image)
