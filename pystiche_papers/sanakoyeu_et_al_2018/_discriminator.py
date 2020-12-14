import contextlib
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from torch import nn

from pystiche import enc
from pystiche_papers.sanakoyeu_et_al_2018._modules import ConvBlock, conv
from pystiche_papers.utils import channel_progression

__all__ = ["MultiScaleDiscriminator", "discriminator"]


class _Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._accuracy = 0.0
        self._accuracies: List[torch.Tensor] = []
        self._record_accuracies = False
        self._real = True

        self.register_forward_hook(type(self)._forward_hook)

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @contextlib.contextmanager
    def record_accuracies(self):
        self._accuracies = []
        self._record_accuracies = True
        try:
            yield lambda: self._recorder(True), lambda: self._recorder(False)
        finally:
            self._record_accuracies = False
            self._accuracy = torch.mean(torch.stack(self._accuracies)).item()

    @contextlib.contextmanager
    def _recorder(self, real: bool):
        self._real = real
        yield

    def _compute_accuracy(self, prediction: torch.Tensor) -> torch.Tensor:
        comparator = torch.ge if self._real else torch.lt
        return torch.mean(comparator(prediction, 0.0).float())

    def _forward_hook(self, input: Any, output: Any) -> None:
        if not self._record_accuracies:
            return

        if isinstance(output, torch.Tensor):
            self._accuracies.append(self._compute_accuracy(output))
            return

        self.compute_accuracies(output)

    def compute_accuracies(self, output: Any) -> None:
        # TODO: add better error message
        raise NotImplementedError


class MultiScaleDiscriminator(_Discriminator):
    r"""Discriminator from :cite:`SKL+2018`."""

    def __init__(
        self,
        discriminator_modules: Sequence[Tuple[str, nn.Module]],
        prediction_modules: Dict[str, nn.Module],
    ):
        super().__init__()
        self.mle = enc.MultiLayerEncoder(discriminator_modules)
        self.predictors = nn.ModuleDict(prediction_modules)

        for name, predictor in self.predictors.items():
            if name not in self.mle:
                # TODO
                raise ValueError

            self.mle.registered_layers.add(name)

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        encs = self.mle(input, self.predictors.keys())
        return {
            name: predictor(enc)
            for (name, predictor), enc in zip(self.predictors.items(), encs)
        }

    def compute_accuracies(self, multi_scale_predictions: Dict[str, torch.Tensor]):
        for prediction in multi_scale_predictions.values():
            self._accuracies.append(self._compute_accuracy(prediction))


def _discriminator_module(in_channels: int, out_channels: int) -> ConvBlock:
    return ConvBlock(
        in_channels, out_channels, kernel_size=5, stride=2, padding=None, act="lrelu",
    )


def _prediction_module(
    in_channels: int, kernel_size: Union[Tuple[int, int], int],
) -> nn.Conv2d:
    r"""Prediction module from :cite:`SKL+2018`.

    This block comprises a convolutional, which is used as an auxiliary classifier to
    capture image details on different scales of the :class:`DiscriminatorEncoder`.

    Args:
        in_channels: Number of channels in the input.
        kernel_size: Size of the convolving kernel.

    """
    return conv(
        in_channels=in_channels,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=None,
    )


def discriminator():
    channels = (3, 128, 128, 256, 512, 512, 1024, 1024)
    discriminator_modules = tuple(
        zip(
            [str(scale) for scale in range(len(channels) - 1)],
            channel_progression(_discriminator_module, channels=channels,),
        )
    )

    prediction_modules = OrderedDict(
        [
            (scale, _prediction_module(in_channels, kernel_size),)
            for scale, in_channels, kernel_size in (
                ("0", 128, 5),
                ("1", 128, 10),
                ("3", 512, 10),
                ("5", 1024, 6),
                ("6", 1024, 3),
            )
        ]
    )

    return MultiScaleDiscriminator(discriminator_modules, prediction_modules)
