from typing import Collection, Optional, Union

from pystiche import loss, ops, pyramid
from pystiche_papers.utils import HyperParameters

from ._utils import hyper_parameters as _hyper_parameters

__all__ = ["image_pyramid"]


def image_pyramid(
    hyper_parameters: Optional[HyperParameters] = None,
    resize_targets: Collection[Union[ops.Operator, loss.MultiOperatorLoss]] = (),
) -> pyramid.ImagePyramid:
    r"""Image pyramid from :cite:`GEB+2017`.

    Args:
        edge_sizes: Edge sizes for each level. Defaults to ``(500, 800)``.
        num_steps: Number of steps for each level. Defaults to  ``(500, 200)``.
        **image_pyramid_kwargs: Optional parameters for the
            :class:`~pystiche.pyramid.ImagePyramid`.

    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters()
    return pyramid.ImagePyramid(
        hyper_parameters.image_pyramid.edge_sizes,
        hyper_parameters.image_pyramid.num_steps,
        resize_targets=resize_targets,
    )
