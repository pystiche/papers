from pystiche.data import DownloadableImageCollection
from pystiche_papers import data


def test_NPRgeneralLicense_smoke():
    _, image = next(iter(data.NPRgeneral()))
    assert isinstance(repr(image.license), str)


def test_NPRgeneral_smoke():
    assert isinstance(data.NPRgeneral(), DownloadableImageCollection)
