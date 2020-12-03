from typing import Any, Optional

from pystiche import pyramid
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["image_pyramid"]


def image_pyramid(
    impl_params: bool = True,
    hyper_parameters: Optional[HyperParameters] = None,
    **image_pyramid_kwargs: Any,
) -> pyramid.OctaveImagePyramid:
    r"""Image pyramid from :cite:`LW2016`.

    Args:
        impl_params: Switch the behavior and hyper-parameters between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <li_wand_2016-impl_params>`.
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.
        image_pyramid_kwargs: Additional options. See
            :class:`~pystiche.pyramid.ImagePyramid` for details.

    .. seealso::

        - :class:`pystiche.pyramid.OctaveImagePyramid`
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    return pyramid.OctaveImagePyramid(
        hyper_parameters.image_pyramid.max_edge_size,
        hyper_parameters.image_pyramid.num_steps,
        num_levels=hyper_parameters.image_pyramid.num_levels,
        min_edge_size=hyper_parameters.image_pyramid.min_edge_size,
        edge=hyper_parameters.image_pyramid.edge,
        **image_pyramid_kwargs,
    )
