import importlib
from os import path

import pytest

from pystiche_papers import data

from .utils import assert_downloads_correctly, assert_is_downloadable, retry

IMAGES_AND_IDS = [
    (image, f"{paper}, {name}")
    for paper in (
        "gatys_ecker_bethge_2016",
        "gatys_et_al_2017",
        "johnson_alahi_li_2016",
        "li_wand_2016",
        "sanakoyeu_et_al_2018",
        "ulyanov_et_al_2016",
    )
    for name, image in importlib.import_module(f"pystiche_papers.{paper}").images()
]
IMAGES_AND_IDS.extend(
    [
        (image, f"{collection}, {name}")
        for collection in ("NPRgeneral",)
        for name, image in getattr(data, collection)()
    ]
)

IMAGE_PARAMETRIZE_KWARGS = dict(zip(("argvalues", "ids"), zip(*IMAGES_AND_IDS)))


def assert_image_is_downloadable(image, **kwargs):
    assert_is_downloadable(image.url, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize("image", **IMAGE_PARAMETRIZE_KWARGS)
def test_image_download_smoke(subtests, image):
    retry(lambda: assert_image_is_downloadable(image), times=2, wait=5.0)


def assert_image_downloads_correctly(image, **kwargs):
    def downloader(url, root):
        image.download(root=root)
        return path.join(root, image.file)

    assert_downloads_correctly(None, image.md5, downloader=downloader, **kwargs)


@pytest.mark.large_download
@pytest.mark.slow
@pytest.mark.parametrize("image", **IMAGE_PARAMETRIZE_KWARGS)
def test_image_download(image):
    assert_image_downloads_correctly(image)
