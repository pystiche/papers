from typing import Any, Sequence, Union

from pystiche import pyramid

__all__ = ["image_pyramid"]


def image_pyramid(
    edge_sizes: Sequence[int] = (500, 800),
    num_steps: Union[int, Sequence[int]] = (500, 200),
    **image_pyramid_kwargs: Any,
) -> pyramid.ImagePyramid:
    return pyramid.ImagePyramid(edge_sizes, num_steps, **image_pyramid_kwargs)
