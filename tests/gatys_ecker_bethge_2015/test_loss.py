from torch.nn.functional import mse_loss

import pytorch_testing_utils as ptu
from pystiche_papers.gatys_ecker_bethge_2015 import loss


def test_gatys_ecker_bethge_2015_content_loss(
    subtests, multi_layer_encoder_with_layer, target_image, input_image
):
    multi_layer_encoder, layer = multi_layer_encoder_with_layer
    encoder = multi_layer_encoder.extract_encoder(layer)
    target_enc = encoder(target_image)
    input_enc = encoder(input_image)

    configs = ((True, "mean", 1.0), (False, "sum", 1.0 / 2.0))
    for impl_params, loss_reduction, score_correction_factor in configs:
        with subtests.test(impl_params=impl_params):
            op = loss.gatys_ecker_bethge_2015_content_loss(
                impl_params=impl_params,
                multi_layer_encoder=multi_layer_encoder,
                layer=layer,
            )
            op.set_target_image(target_image)
            actual = op(input_image)

            score = mse_loss(input_enc, target_enc, reduction=loss_reduction)
            desired = score * score_correction_factor

            assert actual == ptu.approx(desired)
