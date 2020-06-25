import pytest

import torch
from torch import nn

import pytorch_testing_utils as ptu
from pystiche_papers import utils


def test_Identity():
    input = torch.empty(1)
    model = utils.Identity()
    assert model(input) is input


@pytest.fixture(scope="module")
def double_module():
    class Double(nn.Module):
        def forward(self, x):
            return x * 2.0

    return Double()


def test_ResidualBlock(double_module):
    input = torch.tensor(1.0)
    model = utils.ResidualBlock(double_module)
    assert model(input) == ptu.approx(3.0)


def test_ResidualBlock_shortcut(double_module):
    input = torch.tensor(1.0)
    model = utils.ResidualBlock(
        double_module, shortcut=utils.ResidualBlock(double_module)
    )
    assert model(input) == ptu.approx(5.0)
