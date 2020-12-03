import functools
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import kornia
import kornia.augmentation.functional as F
from kornia import augmentation
from kornia.augmentation.utils import _adapted_uniform
from kornia.geometry import warp_perspective

import torch
from torch import nn
from torch.nn.functional import pad

import pystiche
from pystiche.image import extract_image_size
from pystiche.misc import to_2d_arg
from pystiche_papers.utils import pad_size_to_pad

__all__ = ["pre_crop_augmentation", "post_crop_augmentation"]


class AugmentationBase2d(
    pystiche.ComplexObject, kornia.augmentation.AugmentationBase2D
):
    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["p"] = self.p
        if self.return_transform:
            dct["return_transform"] = True
        return dct


def autocast_params(
    fn: Callable[[Any, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]
) -> Callable[[Any, torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
    @functools.wraps(fn)
    def wrapper(
        self: Any, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return fn(
            self,
            input,
            {name: param.to(input.device) for name, param in params.items()},
        )

    return wrapper


def generate_vertices_from_size(batch_size: int, size: Tuple[int, int]) -> torch.Tensor:
    height, width = size
    return (
        torch.tensor(
            ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1),),
            dtype=torch.float,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )


class RandomRescale(AugmentationBase2d):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L63-L67
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L105-L114
    def __init__(
        self,
        factor: Union[Tuple[float, float], float],
        p: float = 50e-2,
        interpolation: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__(p=p, return_transform=False)
        # Due to a bug in the implementation (low == high) the factor has not to be
        # chosen randomly
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L65-L66
        self.factor = to_2d_arg(factor)
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.same_on_batch = True

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = batch_shape
        vert_factor, horz_factor = self.factor

        start_points = generate_vertices_from_size(batch_size, (height, width))
        end_points = generate_vertices_from_size(
            batch_size, (int(height * vert_factor), int(width * horz_factor))
        )
        return dict(start_points=start_points, end_points=end_points,)

    @autocast_params
    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return F.compute_perspective_transformation(input, params)

    @autocast_params
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        transform = self.compute_transformation(input, params)
        vertex = params["end_points"][0, 2, :]
        size = cast(Tuple[int, int], tuple((vertex + 1).int().tolist()[::-1]))
        return warp_perspective(
            input,
            transform,
            size,
            flags=self.interpolation,
            align_corners=self.align_corners,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["factor"] = self.factor
        if self.interpolation != "bilinear":
            dct["interpolation"] = self.interpolation
        if self.same_on_batch:
            dct["same_on_batch"] = True
        if self.align_corners:
            dct["align_corners"] = True
        return dct


def affine_matrix_from_three_points(
    points: torch.Tensor, transformed_points: torch.Tensor
) -> torch.Tensor:
    mat1 = transformed_points
    mat2 = torch.inverse(torch.cat((points, torch.ones(points.size()[0], 1, 3)), dim=1))
    return torch.bmm(mat1, mat2)


class RandomAffine(AugmentationBase2d):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L78-L81
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L170-L180
    def __init__(
        self,
        shift: float,
        p: float = 50e-2,
        interpolation: str = "bilinear",
        same_on_batch: bool = False,
        align_corners: bool = False,
    ):
        super().__init__(p=p, return_transform=False)
        self.shift = shift
        self.interpolation = interpolation
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        points = torch.tensor((((0, 0, 1), (0, 1, 0)),), dtype=torch.float)
        points = points.repeat(input_shape[0], 1, 1)
        shift = _adapted_uniform(
            points.size(), -self.shift, self.shift, same_on_batch=self.same_on_batch
        )

        return {"matrix": affine_matrix_from_three_points(points, points + shift)}

    @autocast_params
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return F.warp_affine(
            input,
            params["matrix"],
            extract_image_size(input),
            flags=self.interpolation,
            align_corners=self.align_corners,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["shift"] = self.shift
        if self.interpolation != "bilinear":
            dct["interpolation"] = self.interpolation
        if self.same_on_batch:
            dct["same_on_batch"] = True
        if self.align_corners:
            dct["align_corners"] = True
        return dct


Range = Union[torch.Tensor, float, Tuple[float, float], List[float]]


def _range_to_bounds(range: Range) -> Tuple[float, float]:
    if isinstance(range, int):
        range = float(range)
    if isinstance(range, float):
        if range < 0.0:
            raise ValueError
        return (-range, range)

    if isinstance(range, torch.Tensor):
        range = torch.flatten(range).tolist()

    if not isinstance(range, (tuple, list)):
        raise TypeError

    if len(range) != 2:
        raise ValueError

    return cast(Tuple[int, int], tuple(float(item) for item in range))


def random_hsv_jitter_generator(
    batch_size: int,
    same_on_batch: bool,
    hue_scale: Range,
    hue_shift: Range,
    saturation_scale: Range,
    saturation_shift: Range,
    value_scale: Range,
    value_shift: Range,
) -> Dict[str, torch.Tensor]:
    def generate(range: Range, shift: bool) -> torch.Tensor:
        low, high = _range_to_bounds(range)
        if not shift:
            low = 1.0 / (1.0 - low)
            high = 1.0 + high
        sample = _adapted_uniform((batch_size,), low, high, same_on_batch=same_on_batch)
        return sample.view(-1, 1, 1, 1)

    parameters = {}
    for parameter in ("hue", "saturation", "value"):
        for shift in (False, True):
            name = f"{parameter}_{'shift' if shift else 'scale'}"
            range = locals()[name]
            parameters[name] = generate(range, shift)
    return parameters


def apply_hsv_jitter(
    input: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    h, s, v = torch.split(kornia.rgb_to_hsv(input), 1, dim=1)
    h = torch.clamp(
        h * params["hue_scale"] + params["hue_shift"],
        (1e-2 * 2.0 * kornia.pi).item(),
        (99e-2 * 2.0 * kornia.pi).item(),
    )
    s = torch.clamp(
        s * params["saturation_scale"] + params["saturation_shift"], 1e-2, 99e-2
    )
    v = torch.clamp(v * params["value_scale"] + params["value_shift"], 1e-2, 99e-2)
    return kornia.hsv_to_rgb(torch.cat((h, s, v), dim=1))


class RandomHSVJitter(AugmentationBase2d):
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L89-L95
    # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L141-L167
    def __init__(
        self,
        hue_scale: Range,
        hue_shift: Range,
        saturation_scale: Range,
        saturation_shift: Range,
        value_scale: Range,
        value_shift: Range,
        p: float = 0.5,
        same_on_batch: bool = False,
    ):
        super().__init__(p=p, return_transform=False)
        self.hue_scale = hue_scale
        self.hue_shift = hue_shift
        self.saturation_scale = saturation_scale
        self.saturation_shift = saturation_shift
        self.value_scale = value_scale
        self.value_shift = value_shift
        self.same_on_batch = same_on_batch

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return random_hsv_jitter_generator(
            input_shape[0],
            self.same_on_batch,
            self.hue_scale,
            self.hue_shift,
            self.saturation_scale,
            self.saturation_shift,
            self.value_scale,
            self.value_shift,
        )

    @autocast_params
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return apply_hsv_jitter(input, params)


class DynamicSizePad2d(pystiche.Module):
    def __init__(
        self,
        transform: nn.Module,
        factor: Union[Tuple[float, float], float],
        mode: str = "constant",
        value: float = 0,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.factor = cast(Tuple[int, int], to_2d_arg(factor))
        self.mode = mode
        self.value = value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_size = extract_image_size(input)
        input = self.add_padding(input, input_size)
        output = self.transform(input)
        return self.remove_padding(output, input_size)

    def add_padding(self, input: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        height, width = size
        vert_factor, horz_factor = self.factor
        pad_size = int(height * vert_factor), int(height * horz_factor)
        return pad(input, pad_size_to_pad(pad_size), mode=self.mode, value=self.value)

    @staticmethod
    def _compute_pad(pad_size: Tuple[int, int]) -> List[int]:
        def split(total: int) -> Tuple[int, int]:
            pre = total // 2
            post = total - pre
            return pre, post

        vert_pad, horz_pad = pad_size
        top_pad, bottom_pad = split(vert_pad)
        left_pad, right_pad = split(horz_pad)
        return [left_pad, right_pad, top_pad, bottom_pad]

    def remove_padding(
        self, input: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        return kornia.center_crop(input, size)

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["factor"] = self.factor
        if self.mode != "constant":
            dct["mode"] = self.mode
        elif self.value != 0:
            dct["mode"] = "constant"
            dct["value"] = self.value
        return dct


def pre_crop_augmentation(
    p: float = 50e-2, same_on_batch: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L27
        # The interpolation mode should be "bicubic", but this isn't supported
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L113
        RandomRescale(factor=80e-2, p=p, interpolation="bilinear", align_corners=True),
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L71
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L82
        DynamicSizePad2d(
            nn.Sequential(
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L28
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L72-L76
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L116-L127
                augmentation.RandomRotation(
                    degrees=0.15 * 90, p=p, same_on_batch=same_on_batch,
                ),
                # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L33
                RandomAffine(shift=5e-2, p=p, same_on_batch=same_on_batch),
            ),
            factor=50e-2,
            mode="reflect",
        ),
    )


def post_crop_augmentation(
    p: float = 50e-2, same_on_batch: bool = True
) -> nn.Sequential:
    return nn.Sequential(
        # Hue scaling is not implemented and thus we set it to 0
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L273-L276
        RandomHSVJitter(
            hue_scale=0e-2,
            hue_shift=5e-2,
            saturation_scale=5e-2,
            saturation_shift=5e-2,
            value_scale=5e-2,
            value_shift=5e-2,
            p=100e-2,
            same_on_batch=same_on_batch,
        ),
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L34
        augmentation.RandomHorizontalFlip(p=p, same_on_batch=same_on_batch),
        # The vertical flip is used with a probability of 0% and thus left out
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/model.py#L272
    )
