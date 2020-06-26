import pytorch_testing_utils as ptu
from torch.nn.functional import mse_loss

from pystiche.image import extract_batch_size
from pystiche_papers.ulyanov_et_al_2016 import loss


def test_ulyanov_et_al_2016_content_loss(
    subtests, multi_layer_encoder_with_layer, batch_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_image = batch_image
    input_image = batch_image.flip((0,))
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)
    batch_size = extract_batch_size(batch_image)

    configs = ((True, batch_size), (False, 1.0))
    for impl_params, extra_batch_size_mean in configs:
        op = loss.ulyanov_et_al_2016_content_loss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            layer=layer,
        )
        op.set_target_image(target_image)
        actual = op(input_image)

        score = mse_loss(input_enc, target_enc)
        desired = score / extra_batch_size_mean

        assert actual == ptu.approx(desired)
