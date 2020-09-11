import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche.data import DownloadableImageCollection


def test_images_smoke():
    assert isinstance(paper.images(), DownloadableImageCollection)
