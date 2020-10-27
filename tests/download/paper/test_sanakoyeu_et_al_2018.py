import pytest

import pystiche_papers.sanakoyeu_et_al_2018 as paper

from ..utils import assert_is_downloadable


@pytest.mark.slow
def test_WikiArt_downloadable(subtests):
    base = paper.WikiArt.BASE_URL
    styles = paper.WikiArt.STYLES
    for style in styles:
        assert_is_downloadable(f"{base}{style}.tar.gz")
