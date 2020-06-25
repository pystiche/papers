from typing import Optional, cast

import torch
from torch import nn

__all__ = ["Identity", "ResidualBlock"]


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.residual = residual

        if shortcut is None:
            shortcut = Identity()
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.residual(x) + self.shortcut(x))
