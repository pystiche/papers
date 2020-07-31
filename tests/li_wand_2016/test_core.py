import pytorch_testing_utils as ptu

import pystiche_papers.li_wand_2016 as paper
from pystiche_papers.li_wand_2016 import utils
from tests._utils import is_callable


def test_li_wand_2016_nst_smoke(subtests, mocker, content_image, style_image):
    mock = mocker.patch(
        "pystiche_papers.li_wand_2016.core.default_image_pyramid_optim_loop"
    )

    paper.li_wand_2016_nst(content_image, style_image)

    args, kwargs = mock.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]
    initial_resize = pyramid[-1].resize_image

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, initial_resize(content_image))

    with subtests.test("style_image"):
        desired_style_image = preprocessor(initial_resize(style_image))
        for op in criterion.style_loss.operators():
            ptu.assert_allclose(op.target_image, desired_style_image)

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.li_wand_2016_perceptual_loss()))

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.li_wand_2016_image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(optimizer, type(utils.li_wand_2016_optimizer(input_image)))

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(utils.li_wand_2016_preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(utils.li_wand_2016_postprocessor()))
