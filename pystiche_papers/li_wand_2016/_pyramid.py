from typing import Collection, Optional, Union

from pystiche import pyramid
from pystiche.loss import MultiOperatorLoss
from pystiche.ops import Operator
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["image_pyramid"]


def image_pyramid(
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
    interpolation_mode: str = "bilinear",
    resize_targets: Collection[Union[Operator, MultiOperatorLoss]] = (),
) -> pyramid.OctaveImagePyramid:
    r"""Image pyramid from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.
        interpolation_mode: Interpolation mode used for the resizing of the images.
            Defaults to ``"bilinear"``.
        resize_targets: Targets for resizing of set images and guides during iteration.
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    return pyramid.OctaveImagePyramid(
        hyper_parameters.image_pyramid.max_edge_size,
        hyper_parameters.image_pyramid.num_steps,
        num_levels=hyper_parameters.image_pyramid.num_levels,
        min_edge_size=hyper_parameters.image_pyramid.min_edge_size,
        edge=hyper_parameters.image_pyramid.edge,
        interpolation_mode=interpolation_mode,
        resize_targets=resize_targets,
    )
