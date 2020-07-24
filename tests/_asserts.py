from os import path
from urllib.request import Request, urlopen

from torchvision.datasets.utils import calculate_md5

from ._utils import get_tmp_dir

__all__ = ["assert_image_is_downloadable", "assert_image_downloads_correctly"]


def assert_image_is_downloadable(image):
    assert_is_downloadable(image.url)


def assert_is_downloadable(url):
    request = Request(url, headers={"User-Agent": "pystiche_papers"}, method="HEAD")
    response = urlopen(request)
    assert response.code == 200, f"Server returned status code {response.code}."


def assert_image_downloads_correctly(image):
    with get_tmp_dir() as root:
        image.download(root=root)

        file = path.join(root, image.file)

        assert path.exists(file), f"File {file} does not exist after download."

        if image.md5 is not None:
            actual = calculate_md5(file)
            desired = image.md5
            msg = (
                f"The actual and desired MD5 hash of the image mismatch: "
                f"{actual} != {desired}"
            )
            assert actual == desired, msg
