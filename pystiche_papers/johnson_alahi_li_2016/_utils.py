from typing import Dict, Optional, Tuple, TypeVar

from torch import nn, optim

from pystiche import enc
from pystiche.image import transforms
from pystiche_papers.utils import Identity

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


def preprocessor(impl_params: bool = True) -> nn.Module:
    r"""Preprocessor from :cite:`JAL2016`.

    Args:
        impl_params: If ``True``, the input is preprocessed for models trained with
            the Caffe framework. If ``False``, the preprocessor performs the identity
            operation.

    .. seealso::

        - :class:`pystiche.image.transforms.CaffePreprocessing`
    """
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/fast_neural_style/preprocess.lua#L57-L62
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/fast_neural_style/DataLoader.lua#L92
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/train.lua#L133
    return transforms.CaffePreprocessing() if impl_params else Identity()


def postprocessor(impl_params: bool = True) -> nn.Module:
    r"""Preprocessor from :cite:`JAL2016`.

    Args:
        impl_params: If ``True``, the input is postprocessed from models trained with
            the Caffe framework. If ``False``, the postprocessor performs the identity
            operation.

    .. seealso::

        - :class:`pystiche.image.transforms.CaffePostprocessing`
    """
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/fast_neural_style/preprocess.lua#L66-L71
    # https://github.com/pmeier/fast-neural-style/blob/813c83441953ead2adb3f65f4cc2d5599d735fa7/fast_neural_style.lua#L89
    return transforms.CaffePostprocessing() if impl_params else Identity()


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
