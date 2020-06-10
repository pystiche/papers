import torch
from torch import optim

from pystiche.enc import MultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing

__all__ = [
    "gatys_et_al_2017_preprocessor",
    "gatys_et_al_2017_postprocessor",
    "gatys_et_al_2017_multi_layer_encoder",
    "gatys_et_al_2017_optimizer",
]


def gatys_et_al_2017_preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def gatys_et_al_2017_postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def gatys_et_al_2017_multi_layer_encoder() -> VGGMultiLayerEncoder:
    return vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def gatys_et_al_2017_optimizer(input_image: torch.Tensor) -> optim.LBFGS:  # type: ignore[name-defined]
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)  # type: ignore[attr-defined]
