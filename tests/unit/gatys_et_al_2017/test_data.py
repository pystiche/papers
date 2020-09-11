import pystiche_papers.gatys_et_al_2017 as paper
from pystiche.data import DownloadableImageCollection


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)
