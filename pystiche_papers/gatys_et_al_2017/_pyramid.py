from typing import Any, Optional

from pystiche import pyramid
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["image_pyramid"]


def image_pyramid(
    hyper_parameters: Optional[HyperParameters] = None,
    **image_pyramid_kwargs: Any,
) -> pyramid.ImagePyramid:
    r"""Image pyramid from :cite:`GEB+2017`.

    Args:
        hyper_parameters: If omitted,
            :func:`~pystiche_papers.gatys_et_al_2017.hyper_parameters` is used.
        **image_pyramid_kwargs: Additional parameters of a
            :class:`pystiche.pyramid.ImagePyramid`.

    .. seealso::

        - :class:`pystiche.pyramid.ImagePyramid`
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()

    return pyramid.ImagePyramid(
        hyper_parameters.image_pyramid.edge_sizes,
        hyper_parameters.image_pyramid.num_steps,
        **image_pyramid_kwargs,
    )
