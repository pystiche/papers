import pytest

from torch.nn.functional import mse_loss

from pystiche_papers import gatys_ecker_bethge_2015 as paper
from pystiche_papers.gatys_ecker_bethge_2015 import loss

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_gatys_ecker_bethge_2015_images_smoke(subtests):
    for name, image in paper.gatys_ecker_bethge_2015_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_gatys_ecker_bethge_2015_images(subtests):
    for name, image in paper.gatys_ecker_bethge_2015_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_gatys_ecker_bethge_2015_content_loss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean", 1.0), (False, "sum", 1.0 / 2.0))
    for impl_params, loss_reduction, score_correction_factor in configs:
        op = loss.gatys_ecker_bethge_2015_content_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layer=layer,
        )
        op.set_target_image(target_image)
        actual = op(input_image)

        score = mse_loss(input_enc, target_enc, reduction=loss_reduction)
        desired = score * score_correction_factor

        assert (actual - desired).abs().max().item() < 1e-6
