from typing import Dict, Optional, Tuple, TypeVar

from torch import nn, optim

from pystiche import enc
from pystiche.image import transforms

__all__ = [
    "_maybe_get_luatorch_param",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


T = TypeVar("T")


def _maybe_get_luatorch_param(
    param_dict: Dict[Tuple[str, bool], T],
    impl_params: bool,
    instance_norm: bool,
    style: Optional[str],
    default: T,
) -> T:
    if style is None or not impl_params:
        return default

    try:
        return param_dict[(style, instance_norm)]
    except KeyError:
        return default


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
