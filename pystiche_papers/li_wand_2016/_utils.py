from typing import Any, Sequence, Tuple, Union

import torch
from torch import optim

import pystiche
from pystiche.enc import MultiLayerEncoder, vgg19_multi_layer_encoder
from pystiche.image.transforms import CaffePostprocessing, CaffePreprocessing
from pystiche.misc import to_2d_arg

__all__ = [
    "extract_normalized_patches2d",
    "preprocessor",
    "postprocessor",
    "multi_layer_encoder",
    "optimizer",
]


class NormalizeUnfoldGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, dim: int, size: int, step: int) -> torch.Tensor:  # type: ignore[override]
        ctx.needs_normalizing = step < size
        if ctx.needs_normalizing:
            normalizer = torch.zeros_like(input)
            item = [slice(None) for _ in range(input.dim())]
            for idx in range(0, normalizer.size()[dim] - size, step):
                item[dim] = slice(idx, idx + size)
                normalizer[item].add_(1.0)

            # clamping to avoid zero division
            ctx.save_for_backward(torch.clamp(normalizer, min=1.0))
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:  # type: ignore[override]
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
    patch_size = to_2d_arg(patch_size)
    stride = to_2d_arg(stride)
    for dim, size, step in zip(range(2, input.dim()), patch_size, stride):
        input = normalize_unfold_grad(input, dim, size, step)
    return pystiche.extract_patches2d(input, patch_size, stride)


def preprocessor() -> CaffePreprocessing:
    return CaffePreprocessing()


def postprocessor() -> CaffePostprocessing:
    return CaffePostprocessing()


def multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_multi_layer_encoder(
        weights="caffe", internal_preprocessing=False, allow_inplace=True
    )


def optimizer(input_image: torch.Tensor) -> optim.LBFGS:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)
