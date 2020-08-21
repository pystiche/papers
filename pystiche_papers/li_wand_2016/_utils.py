from typing import Any, Sequence, Tuple, Union, cast

import torch
from torch import optim

import pystiche
from pystiche import enc, misc
from pystiche.image import transforms

__all__ = [
    "extract_normalized_patches2d",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


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


def preprocessor() -> transforms.CaffePreprocessing:
    return transforms.CaffePreprocessing()


def postprocessor() -> transforms.CaffePostprocessing:
    return transforms.CaffePostprocessing()


def multi_layer_encoder() -> enc.MultiLayerEncoder:
    r"""Multi-layer encoder from :cite:`LW2016`."""
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(  # type: ignore[attr-defined]
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )
    return cast(enc.VGGMultiLayerEncoder, multi_layer_encoder)


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    r"""Optimizer from :cite:`LW2016`.

    Args:
        input_image: Image to be optimized.

    """
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
