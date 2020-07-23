import pytest

import pytorch_testing_utils as ptu

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche_papers.gatys_et_al_2017 import utils
from tests._utils import is_callable


def test_gatys_et_al_2017_nst_smoke(subtests, mocker, content_image, style_image):
    mock = mocker.patch(
        "pystiche_papers.gatys_et_al_2017.core.default_image_pyramid_optim_loop"
    )

    paper.gatys_et_al_2017_nst(content_image, style_image)

    args, kwargs = mock.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(criterion, type(paper.gatys_et_al_2017_perceptual_loss()))

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.gatys_et_al_2017_image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(
            optimizer, type(utils.gatys_et_al_2017_optimizer(input_image))
        )

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(utils.gatys_et_al_2017_preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(utils.gatys_et_al_2017_postprocessor()))


def test_gatys_et_al_2017_guided_nst_smoke(
    subtests, mocker, content_image, style_image, content_guides, style_guides
):
    mock = mocker.patch(
        "pystiche_papers.gatys_et_al_2017.core.default_image_pyramid_optim_loop"
    )
    style_images_and_guides = {
        region: (style_image, guide) for region, guide in style_guides.items()
    }

    paper.gatys_et_al_2017_guided_nst(
        content_image, content_guides, style_images_and_guides
    )

    args, kwargs = mock.call_args
    input_image, criterion, pyramid = args
    get_optimizer = kwargs["get_optimizer"]
    preprocessor = kwargs["preprocessor"]
    postprocessor = kwargs["postprocessor"]

    with subtests.test("input_image"):
        ptu.assert_allclose(input_image, pyramid[-1].resize_image(content_image))

    with subtests.test("criterion"):
        assert isinstance(
            criterion,
            type(paper.gatys_et_al_2017_guided_perceptual_loss(content_guides.keys())),
        )

    with subtests.test("pyramid"):
        assert isinstance(pyramid, type(paper.gatys_et_al_2017_image_pyramid()))

    with subtests.test("optimizer"):
        assert is_callable(get_optimizer)
        optimizer = get_optimizer(input_image)
        assert isinstance(
            optimizer, type(utils.gatys_et_al_2017_optimizer(input_image))
        )

    with subtests.test("preprocessor"):
        assert isinstance(preprocessor, type(utils.gatys_et_al_2017_preprocessor()))

    with subtests.test("postprocessor"):
        assert isinstance(postprocessor, type(utils.gatys_et_al_2017_postprocessor()))


def test_test_gatys_et_al_2017_guided_nst_regions_mismatch(
    content_image, content_guides, style_image, style_guides
):
    style_images_and_guides = {
        region: (style_image, guide)
        for region, guide in tuple(style_guides.items())[:-1]
    }

    with pytest.raises(RuntimeError):
        paper.gatys_et_al_2017_guided_nst(
            content_image, content_guides, style_images_and_guides
        )
