import pytest

from torch.nn.functional import mse_loss

from pystiche.image import extract_batch_size
from pystiche_papers import ulyanov_et_al_2016 as paper
from pystiche_papers.ulyanov_et_al_2016 import loss

from .asserts import assert_image_downloads_correctly, assert_image_is_downloadable


def test_ulyanov_et_al_2016_images_smoke(subtests):
    for name, image in paper.ulyanov_et_al_2016_images():
        with subtests.test(name=name):
            assert_image_is_downloadable(image)


@pytest.mark.large_download
@pytest.mark.slow
def test_ulyanov_et_al_2016_images(subtests):
    for name, image in paper.ulyanov_et_al_2016_images():
        with subtests.test(name=name):
            assert_image_downloads_correctly(image)


def test_ulyanov_et_al_2016_content_loss(
    subtests, multi_layer_encoder_with_layer, multi_target_image, multi_input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(multi_target_image)
    input_enc = encoder(multi_input_image)
    batch_size = extract_batch_size(multi_target_image)

    configs = ((True, batch_size), (False, 1.0))
    for impl_params, extra_batch_size_mean in configs:
        op = loss.ulyanov_et_al_2016_content_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layer=layer,
        )
        op.set_target_image(multi_target_image)
        actual = op(multi_input_image)

        score = mse_loss(input_enc, target_enc)
        desired = score / extra_batch_size_mean
        assert (actual - desired).abs().max().item() < 1e-6
