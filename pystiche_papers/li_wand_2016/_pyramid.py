from typing import Any, Optional, Sequence, Union

from pystiche import pyramid

__all__ = ["image_pyramid"]


def image_pyramid(
    impl_params: bool = True,
    max_edge_size: int = 384,
    num_steps: Optional[Union[int, Sequence[int]]] = None,
    num_levels: Optional[int] = None,
    min_edge_size: int = 64,
    edge: Union[str, Sequence[str]] = "long",
    **octave_image_pyramid_kwargs: Any,
) -> pyramid.OctaveImagePyramid:
    r"""Image pyramid from :cite:`LW2016`.

    Args:
        impl_params: If ``True``, uses the parameters used in the reference
            implementation of the original authors rather than what is described in
            the paper.
        max_edge_size: Maximum edge size. Defaults to ``384``.
        num_steps: Number of steps for each level. If omitted, the number is
            determined with respect to ``impl_params``. Defaults to ``100`` if
            ``impl_params is True`` otherwise ``200``.
        num_levels: Optional number of levels. If omitted, the number is determined by
            the number of steps of factor two between ``max_edge_size`` and
            ``min_edge_size`` or with respect to ``impl_params``. Defaults to ``3`` if
            ``impl_params is True`` otherwise the number is determined by
            the number of steps of factor two between ``max_edge_size`` and
            ``min_edge_size``.
        min_edge_size: Minimum edge size for the automatic calculation of
            ``num_levels``. Defaults to ``64``.
        edge: Corresponding edge to the edge size for each level. Can be ``"short"`` or
            ``"long"``. If sequence of ``str`` its length has to match the length of
            ``edge_sizes``. Defaults to ``"long"``.
        **octave_image_pyramid_kwargs: Optional parameters for the
            :class:`~pystiche.pyramid.OctaveImagePyramid`.

    """
    if num_steps is None:
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L44
        num_steps = 100 if impl_params else 200

    if num_levels is None:
        # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L43
        num_levels = 3 if impl_params else None

    return pyramid.OctaveImagePyramid(
        max_edge_size,
        num_steps,
        num_levels=num_levels,
        min_edge_size=min_edge_size,
        edge=edge,
        **octave_image_pyramid_kwargs,
    )
