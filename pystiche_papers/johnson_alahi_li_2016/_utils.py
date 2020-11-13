from torch import nn, optim

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import HyperParameters

__all__ = [
    "hyper_parameters",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


def hyper_parameters() -> HyperParameters:
    r"""Hyper parameters from :cite:`JAL2016`."""
    return HyperParameters(
        content_loss=HyperParameters(
            layer="relu2_2",
            # The paper reports no score weight so we go with the default value of the
            # implementation instead
            # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L36
            score_weight=1e0,
        ),
        style_loss=HyperParameters(
            layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3"),
            layer_weights="sum",
            # The paper reports no style score weight so we go with the default value
            # of the implementation instead
            # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L43
            score_weight=5e0,
        ),
        regularization=HyperParameters(
            # The paper reports a range of regularization score weights so we go with
            # the default value of the implementation instead
            # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L33
            score_weight=1e-6,
        ),
        content_transform=HyperParameters(edge_size=256),
        style_transform=HyperParameters(edge_size=256, edge="long"),
        batch_sampler=HyperParameters(num_batches=40000, batch_size=4),
    )


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def multi_layer_encoder(impl_params: bool = True,) -> enc.VGGMultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`JAL2016`.

    Args:
        impl_params: If ``True``, the necessary preprocessing of the images is
            performed internally.
    """
    return enc.vgg16_multi_layer_encoder(
        weights="caffe", internal_preprocessing=not impl_params, allow_inplace=True
    )


def optimizer(transformer: nn.Module) -> optim.Adam:
    r"""Optimizer from :cite:`JAL2016`.

    Args:
        transformer: Transformer to be optimized.

    """
    return optim.Adam(transformer.parameters(), lr=1e-3)
