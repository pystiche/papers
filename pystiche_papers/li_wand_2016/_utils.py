import itertools
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import torch
from torch import optim

import pystiche
from pystiche import enc, misc, ops
from pystiche.image import extract_image_size, transforms
from pystiche.image.transforms.functional import crop
from pystiche_papers.utils import HyperParameters

__all__ = [
    "hyper_parameters",
    "extract_normalized_patches2d",
    "target_transforms",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


def hyper_parameters(impl_params: bool = True) -> HyperParameters:
    r"""Hyper parameters from :cite:`LW2016`.

    Args:
        impl_params: Switch the hyper parameters and behavior between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <table-li_wand_2016-impl_params>`.
    """
    return HyperParameters(
        content_loss=HyperParameters(
            layer="relu4_2",
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L58
            score_weight=2e1 if impl_params else 1e0,
        ),
        target_transforms=HyperParameters(
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L52
            num_scale_steps=0 if impl_params else 3,
            scale_step_width=5e-2,
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L51
            num_rotate_steps=0 if impl_params else 2,
            rotate_step_width=7.5,
        ),
        style_loss=HyperParameters(
            layers=("relu3_1", "relu4_1"),
            layer_weights="sum",
            patch_size=3,
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L53
            stride=2 if impl_params else 1,
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L49
            score_weight=1e-4 if impl_params else 1e0,
        ),
        regularization=HyperParameters(score_weight=1e-3),
        image_pyramid=HyperParameters(
            max_edge_size=384,
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L44
            num_steps=100 if impl_params else 200,
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/cnnmrf.lua#L43
            num_levels=3 if impl_params else None,
            min_edge_size=64,
            edge="long",
        ),
        nst=HyperParameters(starting_point="content" if impl_params else "random"),
    )


_hyper_parameters = hyper_parameters


class NormalizeUnfoldGrad(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, input: torch.Tensor, dim: int, size: int, step: int
    ) -> torch.Tensor:
        ctx.needs_normalizing = step < size
        if ctx.needs_normalizing:
            normalizer = torch.zeros_like(input)
            item = [slice(None) for _ in range(input.dim())]
            for idx in range(0, normalizer.size()[dim] - size + 1, step):
                item[dim] = slice(idx, idx + size)
                normalizer[item].add_(1.0)

            # clamping to avoid zero division
            ctx.save_for_backward(torch.clamp(normalizer, min=1.0))
        return input

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        if ctx.needs_normalizing:
            (normalizer,) = ctx.saved_tensors
            grad_input = grad_output / normalizer
        else:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


normalize_unfold_grad = NormalizeUnfoldGrad.apply


def extract_normalized_patches2d(
    input: torch.Tensor,
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> torch.Tensor:
    r"""Extract 2-dimensional patches from the input with normalized gradient.

    If ``stride >= patch_size``, this behaves just like
    :func:`pystiche.extract_patches2d`. Otherwise, the gradient of the input is
    normalized such that every value is divided by the number of patches it appears in.

    Examples:
        >>> import torch
        >>> import pystiche
        >>> input = torch.ones(1, 1, 4, 4).requires_grad_(True)
        >>> target = torch.zeros(1, 1, 4, 4).detach()
        >>> # without normalized gradient
        >>> input_patches = pystiche.extract_patches2d(
        ...    input, patch_size=2, stride=1
        ... )
        >>> target_patches = pystiche.extract_patches2d(
        ...     target, patch_size=2, stride=1
        ... )
        >>> loss = 0.5 * torch.sum((input_patches - target_patches) ** 2.0)
        >>> loss.backward()
        >>> input.grad
        tensor([[[[1., 2., 2., 1.],
                  [2., 4., 4., 2.],
                  [2., 4., 4., 2.],
                  [1., 2., 2., 1.]]]])

        >>> import torch
        >>> import pystiche
        >>> import pystiche_papers.li_wand_2016 as paper
        >>> input = torch.ones(1, 1, 4, 4).requires_grad_(True)
        >>> target = torch.zeros(1, 1, 4, 4).detach()
        >>> # with normalized gradient
        >>> input_patches = paper.extract_normalized_patches2d(
        ...    input, patch_size=2, stride=1
        ... )
        >>> target_patches = pystiche.extract_patches2d(
        ...     target, patch_size=2, stride=1
        ... )
        >>> loss = 0.5 * torch.sum((input_patches - target_patches) ** 2.0)
        >>> loss.backward()
        >>> input.grad
        tensor([[[[1., 1., 1., 1.],
                  [1., 1., 1., 1.],
                  [1., 1., 1., 1.],
                  [1., 1., 1., 1.]]]])

    Args:
        input: Input tensor of shape :math:`B \times C \times H \times W`
        patch_size: Patch size
        stride: Stride
    """
    patch_size = misc.to_2d_arg(patch_size)
    stride = misc.to_2d_arg(stride)
    for dim, size, step in zip(range(2, input.dim()), patch_size, stride):
        input = normalize_unfold_grad(input, dim, size, step)
    return pystiche.extract_patches2d(input, patch_size, stride)


# Right now, this an (almost) exact port from the source. It needs to be refactored.
# It does the following for a positive alpha (angle):
# - create the vertices of the image
# - rotate these vertices clockwise (!)
# - find the intersection of the old vertical left edge with the new "horizontal"
#   bottom edge
# - find the intersection of the old vertical right with the horizontal extension of
#   the first intersection
# - Use the second intersection as bottom right vertex
# - Use the the difference of image size and bottom right corner as top left vertex
# https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/mylib/helper.lua#L74-L139
def _computeBB(width: int, height: int, alpha: float) -> Tuple[int, int, int, int]:
    x1 = 1
    y1 = 1
    x2 = width
    y2 = 1
    x3 = width
    y3 = height
    x4 = 1
    y4 = height
    x0 = width / 2
    y0 = height / 2

    x1r = x0 + (x1 - x0) * math.cos(alpha) + (y1 - y0) * math.sin(alpha)
    y1r = y0 - (x1 - x0) * math.sin(alpha) + (y1 - y0) * math.cos(alpha)

    x2r = x0 + (x2 - x0) * math.cos(alpha) + (y2 - y0) * math.sin(alpha)
    y2r = y0 - (x2 - x0) * math.sin(alpha) + (y2 - y0) * math.cos(alpha)

    x3r = x0 + (x3 - x0) * math.cos(alpha) + (y3 - y0) * math.sin(alpha)
    y3r = y0 - (x3 - x0) * math.sin(alpha) + (y3 - y0) * math.cos(alpha)

    x4r = x0 + (x4 - x0) * math.cos(alpha) + (y4 - y0) * math.sin(alpha)
    y4r = y0 - (x4 - x0) * math.sin(alpha) + (y4 - y0) * math.cos(alpha)

    if alpha > 0:
        px1 = (
            (x1 * y4 - y1 * x4) * (x1r - x2r) - (x1 - x4) * (x1r * y2r - y1r * x2r)
        ) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
        py1 = (
            (x1 * y4 - y1 * x4) * (y1r - y2r) - (y1 - y4) * (x1r * y2r - y1r * x2r)
        ) / ((x1 - x4) * (y1r - y2r) - (y1 - y4) * (x1r - x2r))
        px2 = px1 + 1
        py2 = py1

        qx = (
            (px1 * py2 - py1 * px2) * (x2r - x3r)
            - (px1 - px2) * (x2r * y3r - y2r * x3r)
        ) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))
        qy = (
            (px1 * py2 - py1 * px2) * (y2r - y3r)
            - (py1 - py2) * (x2r * y3r - y2r * x3r)
        ) / ((px1 - px2) * (y2r - y3r) - (py1 - py2) * (x2r - x3r))

        min_x = width - qx
        min_y = qy
        max_x = qx
        max_y = height - qy

    elif alpha < 0:
        px1 = (
            (x2 * y3 - y2 * x3) * (x1r - x2r) - (x2 - x3) * (x1r * y2r - y1r * x2r)
        ) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
        py1 = (
            (x2 * y3 - y1 * x3) * (y1r - y2r) - (y2 - y3) * (x1r * y2r - y1r * x2r)
        ) / ((x2 - x3) * (y1r - y2r) - (y2 - y3) * (x1r - x2r))
        px2 = px1 - 1
        py2 = py1

        qx = (
            (px1 * py2 - py1 * px2) * (x1r - x4r)
            - (px1 - px2) * (x1r * y4r - y1r * x4r)
        ) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))
        qy = (
            (px1 * py2 - py1 * px2) * (y1r - y4r)
            - (py1 - py2) * (x1r * y4r - y1r * x4r)
        ) / ((px1 - px2) * (y1r - y4r) - (py1 - py2) * (x1r - x4r))
        min_x = qx
        min_y = qy
        max_x = width - min_x
        max_y = height - min_y
    else:
        min_x = x1
        min_y = y1
        max_x = x2
        max_y = y3

    return (
        max(math.floor(min_x), 1),
        max(math.floor(min_y), 1),
        # The clamping to the maximum height and width is not part of the original
        # implementation.
        min(math.floor(max_x), width),
        min(math.floor(max_y), height),
    )


class ValidCropAfterRotate(transforms.Transform):
    def __init__(self, angle: float, clockwise: bool = False):
        super().__init__()
        self.angle = angle
        self.clockwise = clockwise

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        origin, size = self._compute_crop(image)
        return cast(torch.Tensor, crop(image, origin, size))

    def _compute_crop(
        self, image: torch.Tensor
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        height, width = extract_image_size(image)
        alpha = math.radians(self.angle)
        if not self.clockwise:
            alpha *= -1
        bounding_box = _computeBB(width, height, alpha)
        origin = (bounding_box[1], bounding_box[0])
        size = (bounding_box[3] - bounding_box[1], bounding_box[2] - bounding_box[0])
        return origin, size

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["angle"] = f"{self.angle}°"
        if self.clockwise:
            dct["clockwise"] = self.clockwise
        return dct


def target_transforms(
    impl_params: bool = True, hyper_parameters: Optional[HyperParameters] = None,
) -> Sequence[transforms.Transform]:
    r"""MRF target transformations from :cite:`LW2016`.

    Args:
        impl_params: Switch the hyper parameters and behavior between the reference
            implementation of the original authors and what is described in the paper.
            For details see :ref:`here <table-li_wand_2016-impl_params>`.
        hyper_parameters: Hyper parameters. If omitted,
            :func:`~pystiche_papers.li_wand_2016.hyper_parameters` is used.

    .. seealso::

        - :meth:`pystiche.ops.MRFOperator.scale_and_rotate_transforms`
    """
    if hyper_parameters is None:
        hyper_parameters = _hyper_parameters(impl_params=impl_params)

    if not impl_params:
        return ops.MRFOperator.scale_and_rotate_transforms(
            **hyper_parameters.target_transforms
        )

    def symrange(steps: int) -> range:
        return range(-steps, steps + 1)

    scaling_factors = [
        1.0 + (base * hyper_parameters.target_transforms.scale_step_width)
        for base in symrange(hyper_parameters.target_transforms.num_scale_steps)
    ]
    rotation_angles = [
        base * hyper_parameters.target_transforms.rotate_step_width
        for base in symrange(hyper_parameters.target_transforms.num_rotate_steps)
    ]

    transforms_ = []
    for scaling_factor, rotation_angle in itertools.product(
        scaling_factors, rotation_angles
    ):
        transforms_.append(
            transforms.ComposedTransform(
                transforms.RotateMotif(rotation_angle),
                ValidCropAfterRotate(rotation_angle),
                transforms.Rescale(scaling_factor),
            )
        )
    return transforms_


def preprocessor() -> transforms.CaffePreprocessing:
    r"""Preprocessor from :cite:`LW2016`."""
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    r"""Postprocessor from :cite:`LW2016`."""
    return transforms.CaffePostprocessing()


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`LW2016`."""
    return enc.vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""Optimizer from :cite:`LW2016`.

    Args:
        input_image: Image to be optimized.

    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
