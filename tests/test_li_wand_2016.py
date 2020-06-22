import pytest

from torch.nn.functional import mse_loss

from pystiche_papers import li_wand_2016 as paper
from pystiche_papers.li_wand_2016 import loss

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_ulyanov_et_al_2016_images_smoke(subtests):
    for name, image in paper.li_wand_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_li_wand_2016_images(subtests):
    for name, image in paper.li_wand_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_li_wand_2016_content_loss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean", 20.0), (False, "sum", 1.0))
    for impl_params, loss_reduction, score_weight in configs:
        op = loss.li_wand_2016_content_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layer=layer,
        )
        op.set_target_image(target_image)
        actual = op(input_image)

        score = mse_loss(input_enc, target_enc, reduction=loss_reduction)
        desired = score * score_weight
        assert (actual - desired).abs().max().item() < 1e-6
