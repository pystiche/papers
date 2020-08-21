import functools

from tests import mocks

__all__ = ["make_paper_mock_target"]

PAPER = "sanakoyeu_et_al_2018"

make_paper_mock_target = functools.partial(mocks.make_mock_target, PAPER)
