from pystiche.image.transforms import CaffePreprocessing
from pystiche_papers.gatys_et_al_2017 import utils


def test_gatys_et_al_2017_preprocessor():
    assert isinstance(utils.gatys_et_al_2017_preprocessor(), CaffePreprocessing)
