import pytest

from torch.nn.functional import mse_loss

from pystiche_papers.gatys_et_al_2017 import loss


@pytest.fixture
def setup(multi_layer_encoder_with_layer, target_image, input_image):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)
    return multi_layer_encoder, layer, target_enc, input_enc


def test_gatys_et_al_2017_content_loss(setup, target_image, input_image):
    multi_layer_encoder, layer, target_enc, input_enc = setup

    op = loss.gatys_et_al_2017_content_loss(
        multi_layer_encoder=multi_layer_encoder, layer=layer,
    )
    op.set_target_image(target_image)
    actual = op(input_image)

    desired = mse_loss(input_enc, target_enc)

    assert (actual - desired).abs().max().item() < 1e-6
