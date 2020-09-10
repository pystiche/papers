from typing import Any, Dict, List, Tuple, Union, cast

import kornia
import kornia.augmentation.functional as F
from kornia.augmentation.random_generator import _adapted_uniform, random_prob_generator

import torch
from torch import nn
from torch.nn.functional import pad

import pystiche
from pystiche.image import extract_image_size
from pystiche.misc import to_2d_arg

__all__ = ["augmentation"]


class AugmentationBase(pystiche.ComplexObject, kornia.augmentation.AugmentationBase):
    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        if self.return_transform:
            dct["return_transform"] = True
        return dct


class RandomRescale(AugmentationBase):
    def __init__(
        self,
        factor: Union[Tuple[float, float], float],
        probability: float = 50e-2,
        interpolation: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__(return_transform=False)
        # https://github.com/pmeier/adaptive-style-transfer/blob/07a3b3fcb2eeed2bf9a22a9de59c0aea7de44181/img_augm.py#L65-L66 # noqa: W505
        # due to a bug we don't need to randomly chose the factor but only if a
        # scaling is happening
        self.factor = to_2d_arg(factor)
        self.probability = probability
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.same_on_batch = True

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        return cast(
            Dict[str, torch.Tensor],
            random_prob_generator(1, self.probability, self.same_on_batch),
        )

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise RuntimeError

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not params["batch_prob"].item():
            return input

        height, width = extract_image_size(input)
        factor_vert, factor_horz = self.factor
        size = (int(height * factor_vert), width * factor_horz)
        return cast(
            torch.Tensor,
            kornia.resize(
                input,
                size,
                interpolation=self.interpolation,
                align_corners=self.align_corners,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["factor"] = self.factor
        dct["probability"] = f"{self.probability:.1%}"
        if self.interpolation != "bilinear":
            dct["interpolation"] = self.interpolation
        if self.same_on_batch:
            dct["same_on_batch"] = True
        if self.align_corners:
            dct["align_corners"] = True
        return dct


class RandomRotation(pystiche.ComplexObject, kornia.augmentation.RandomRotation):
    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float], List[float]],
        probability: float = 50e-2,
        interpolation: str = "bilinear",
        **kwargs: Any,
    ) -> None:
        super().__init__(degrees, interpolation=interpolation, **kwargs)
        self.probability = probability

    def generate_parameters(self, batch_shape: torch.Size) -> Dict[str, torch.Tensor]:
        batch_params = cast(
            Dict[str, torch.Tensor],
            random_prob_generator(batch_shape[0], self.probability, self.same_on_batch),
        )
        params = cast(Dict[str, torch.Tensor], super().generate_parameters(batch_shape))
        params["degrees"][batch_params["batch_prob"]] = 0.0
        return params

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["degrees"] = self.degrees
        dct["probability"] = f"{self.probability:.1%}"
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
    mat2 = torch.inverse(torch.cat((points, torch.ones(points.size()[0], 1, 3))))
    return torch.bmm(mat1, mat2)


class RandomAffine(AugmentationBase):
    def __init__(
        self,
        shift: float,
        probability: float = 50e-2,
        interpolation: str = "bilinear",
        same_on_batch: bool = False,
        align_corners: bool = False,
    ):
        super().__init__(return_transform=False)
        self.shift = shift
        self.probability = probability
        self.interpolation = interpolation
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        batch_size = input_shape[0]
        batch_params = cast(
            Dict[str, torch.Tensor],
            random_prob_generator(batch_size, self.probability, self.same_on_batch),
        )

        points = torch.tensor((((0, 0, 1), (0, 1, 0)),), dtype=torch.float).repeat(
            batch_size, 1, 1
        )
        shift = _adapted_uniform(
            points.size(), -self.shift, self.shift, same_on_batch=self.same_on_batch
        )
        shift[~batch_params["batch_prob"], ...] = 0.0

        return {"matrix": affine_matrix_from_three_points(points, points + shift)}

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            F.warp_affine(
                input,
                params["matrix"],
                extract_image_size(input),
                flags=self.interpolation,
                align_corners=self.align_corners,
            ),
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["shift"] = self.shift
        dct["probability"] = f"{self.probability:.1%}"
        if self.interpolation != "bilinear":
            dct["interpolation"] = self.interpolation
        if self.same_on_batch:
            dct["same_on_batch"] = True
        if self.align_corners:
            dct["align_corners"] = True
        return dct


def _adapted_uniform_int(
    shape: Union[Tuple, torch.Size],
    low: Union[float, torch.Tensor],
    high: Union[float, torch.Tensor],
    same_on_batch: bool = False,
) -> torch.Tensor:
    return cast(
        torch.Tensor, _adapted_uniform(shape, low, high + 1 - 1e-6, same_on_batch).int()
    )


class RandomCrop(AugmentationBase):
    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        interpolation: str = "bilinear",
        return_transform: bool = False,
        same_on_batch: bool = False,
        align_corners: bool = False,
    ) -> None:
        super().__init__(return_transform=return_transform)
        self.size = cast(Tuple[int, int], to_2d_arg(size))
        self.interpolation = interpolation
        self.same_on_batch = same_on_batch
        self.align_corners = align_corners

    def generate_parameters(self, input_shape: torch.Size) -> Dict[str, torch.Tensor]:
        batch_size = input_shape[0]
        image_size = cast(Tuple[int, int], input_shape[-2:])
        anchors = self.generate_anchors(
            batch_size, image_size, self.size, self.same_on_batch
        )
        dst = self.generate_vertices_from_size(batch_size, self.size)
        src = self.clamp_vertices_to_size(anchors + dst, image_size)
        return dict(
            src=src,
            dst=dst,
            interpolation=torch.tensor(kornia.Resample.get(self.interpolation).value),
            align_corners=torch.tensor(self.align_corners),
        )

    def compute_transformation(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return cast(torch.Tensor, F.compute_crop_transformation(input, params))

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return cast(torch.Tensor, F.apply_crop(input, params))

    @staticmethod
    def generate_anchors(
        batch_size: int,
        image_size: Tuple[int, int],
        crop_size: Tuple[int, int],
        same_on_batch: bool,
    ) -> torch.Tensor:
        def generate_single_dim_anchor(
            batch_size: int, image_length: int, crop_length: int, same_on_batch: bool
        ) -> torch.Tensor:
            diff = image_length - crop_length
            if diff <= 0:
                return torch.zeros((batch_size,), dtype=torch.int)
            else:
                return _adapted_uniform_int((batch_size,), 0, diff, same_on_batch)

        single_dim_anchors = [
            generate_single_dim_anchor(
                batch_size, image_length, crop_length, same_on_batch
            )
            for image_length, crop_length in zip(image_size, crop_size)
        ]
        return torch.stack(single_dim_anchors, dim=1,).unsqueeze(1).repeat(1, 4, 1)

    @staticmethod
    def generate_vertices_from_size(
        batch_size: int, size: Tuple[int, int]
    ) -> torch.Tensor:
        height, width = size
        return (
            torch.tensor(
                ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1),),
                dtype=torch.int,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    @staticmethod
    def clamp_vertices_to_size(
        vertices: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        horz, vert = vertices.split(1, dim=2)
        height, width = size
        return torch.cat(
            (torch.clamp(horz, 0, width - 1), torch.clamp(vert, 0, height - 1),), dim=2,
        )

    def _properties(self) -> Dict[str, Any]:
        dct = super()._properties()
        dct["size"] = self.size
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
        value = cast(
            torch.Tensor,
            _adapted_uniform((batch_size,), low, high, same_on_batch=same_on_batch),
        )
        return value.view(-1, 1, 1, 1)

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
        1e-2 * 2.0 * kornia.pi,
        99e-2 * 2.0 * kornia.pi,
    )
    s = torch.clamp(
        h * params["saturation_scale"] + params["saturation_shift"], 1e-2, 99e-2,
    )
    v = torch.clamp(h * params["value_scale"] + params["value_shift"], 1e-2, 99e-2,)
    return cast(torch.Tensor, kornia.hsv_to_rgb(torch.cat((h, s, v), dim=1)))


class RandomHSVJitter(AugmentationBase):
    def __init__(
        self,
        hue_scale: Range,
        hue_shift: Range,
        saturation_scale: Range,
        saturation_shift: Range,
        value_scale: Range,
        value_shift: Range,
        same_on_batch: bool = False,
    ):
        super().__init__(return_transform=False)
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

    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return apply_hsv_jitter(input, params)


def compute_relative_pad_size(
    image: torch.Tensor, factor: Tuple[float, float]
) -> Tuple[int, int]:
    height, width = extract_image_size(image)
    vert_factor, horz_factor = factor
    return int(height * vert_factor), int(height * horz_factor)


class DynamicSizeReflectionPad2d(nn.Module):
    def __init__(self, factor: Union[Tuple[float, float], float]):
        super().__init__()
        self.factor = cast(Tuple[int, int], to_2d_arg(factor))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        vert_pad, horz_pad = compute_relative_pad_size(input, self.factor)
        pad_ = [horz_pad, horz_pad, vert_pad, vert_pad]
        return pad(input, pad_, mode="reflect")


class RemoveDynamicSizePadding(nn.Module):
    def __init__(self, factor: Union[Tuple[float, float], float]):
        super().__init__()
        self.factor = to_2d_arg(factor)

    @property
    def inverse_factor(self) -> Tuple[float, float]:
        return cast(
            Tuple[float, float], tuple(factor / (1 + factor) for factor in self.factor)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        height, width = extract_image_size(input)
        vert_pad, horz_pad = compute_relative_pad_size(input, self.inverse_factor)
        crop_size = (height - vert_pad, width - horz_pad)
        return cast(torch.Tensor, kornia.center_crop(input, crop_size))


def augmentation(
    size: Union[Tuple[int, int], int] = (768, 768),
    probability: float = 50e-2,
    same_on_batch: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        RandomRescale(factor=80e-2, probability=probability, interpolation="bicubic"),
        DynamicSizeReflectionPad2d(factor=25e-2),
        RandomRotation(
            degrees=0.15 * 90, probability=probability, same_on_batch=same_on_batch
        ),
        RandomAffine(5e-2, probability=probability, same_on_batch=same_on_batch),
        RemoveDynamicSizePadding(factor=25e-2),
        RandomCrop(size, same_on_batch=same_on_batch),
        RandomHSVJitter(
            hue_scale=0e-2,
            hue_shift=5e-2,
            saturation_scale=5e-2,
            saturation_shift=5e-2,
            value_scale=5e-2,
            value_shift=5e-2,
            same_on_batch=same_on_batch,
        ),
        kornia.augmentation.RandomHorizontalFlip(
            p=probability, same_on_batch=same_on_batch
        ),
    )
