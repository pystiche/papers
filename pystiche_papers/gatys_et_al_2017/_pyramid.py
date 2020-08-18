from typing import Any, Sequence, Union

from pystiche import pyramid

__all__ = ["image_pyramid"]


def image_pyramid(
    edge_sizes: Sequence[int] = (500, 800),
    num_steps: Union[int, Sequence[int]] = (500, 200),
    **image_pyramid_kwargs: Any,
) -> pyramid.ImagePyramid:
    r"""Image pyramid from :cite:`GEB+2017`.

    Args:
        edge_sizes: Edge sizes for each level. Defaults to ``(500, 800)``.
        num_steps: Number of steps for each level. Defaults to  ``(500, 200)``.
        **image_pyramid_kwargs: Optional parameters for the
            :class:`~pystiche.pyramid.ImagePyramid`.

    """
    return pyramid.ImagePyramid(edge_sizes, num_steps, **image_pyramid_kwargs)
