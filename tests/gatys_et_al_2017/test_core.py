import pytorch_testing_utils as ptu

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche_papers.gatys_et_al_2017 import utils


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
        assert type(pyramid) is type(  # noqa: E721
            paper.gatys_et_al_2017_image_pyramid()
        )

    with subtests.test("optimizer"):
        assert hasattr(get_optimizer, "__call__")
        optimizer = get_optimizer(input_image)
        assert type(optimizer) is type(  # noqa: E721
            utils.gatys_et_al_2017_optimizer(input_image)
        )

    with subtests.test("preprocessor"):
        assert type(preprocessor) is type(  # noqa: E721
            utils.gatys_et_al_2017_preprocessor()
        )

    with subtests.test("postprocessor"):
        assert type(postprocessor) is type(  # noqa: E721
            utils.gatys_et_al_2017_postprocessor()
        )
