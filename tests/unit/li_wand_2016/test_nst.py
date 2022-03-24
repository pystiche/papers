import pytest

from tests import mocks, utils
from tests.utils import is_callable

import pytorch_testing_utils as ptu

import pystiche_papers.li_wand_2016 as paper
from pystiche.image import extract_image_size
from pystiche.misc import get_input_image


@pytest.mark.slow
def test_li_wand_2016_nst_smoke(subtests, mocker, content_image, style_image):
    spy = mocker.patch(
        mocks.make_mock_target("li_wand_2016", "_nst", "misc", "get_input_image"),
        wraps=get_input_image,
    )
    mock = mocker.patch(
        mocks.make_mock_target(
            "li_wand_2016", "_nst", "optim", "default_image_pyramid_optim_loop"
        )
    )

    hyper_parameters = paper.hyper_parameters()

    paper.nst(content_image, style_image)

    args, kwargs = mock.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]
    initial_resize = pyramid[-1].resize_image

    with subtests.test("input_image"):
        args = utils.call_args_to_namespace(spy.call_args, get_input_image)
        assert args.starting_point == hyper_parameters.nst.starting_point
        assert extract_image_size(args.content_image) == extract_image_size(
            initial_resize(content_image)
        )

    with subtests.test("style_image"):
        desired_style_image = preprocessor(initial_resize(style_image))
        for op in criterion.style_loss.operators():
            ptu.assert_allclose(op.target_image, desired_style_image)

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.perceptual_loss()))

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(optimizer, type(paper.optimizer(input_image)))

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(paper.preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(paper.postprocessor()))
