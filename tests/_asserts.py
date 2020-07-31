from os import path
from urllib.request import Request, urlopen

from torchvision.datasets.utils import calculate_md5

from ._utils import get_tempdir

__all__ = [
    "assert_is_downloadable",
    "assert_downloads_correctly",
    "assert_image_is_downloadable",
    "assert_image_downloads_correctly",
]

USER_AGENT = "pystiche_papers/test_suite"


def assert_response_ok(response):
    assert response.code == 200, f"Server returned status code {response.code}."


def assert_is_downloadable(url):
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    response = urlopen(request)
    assert_response_ok(response)


def default_downloader(url, root):
    file = path.join(root, path.basename(url))
    with open(file, "wb") as fh:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request) as response:
            assert_response_ok(response)

            fh.write(response.read())
    return file


def assert_downloads_correctly(url, md5=None, downloader=default_downloader):
    with get_tempdir() as root:
        file = downloader(url, root)
        assert path.exists(file), f"File {file} does not exist after download."

        if md5 is not None:
            actual = calculate_md5(file)
            desired = md5
            assert actual == desired, (
                f"The actual and desired MD5 hash of the image mismatch: "
                f"{actual} != {desired}"
            )


def assert_image_is_downloadable(image):
    assert_is_downloadable(image.url)


def assert_image_downloads_correctly(image):
    def downloader(url, root):
        image.download(root=root)
        return path.join(root, image.file)

    assert_downloads_correctly(None, md5=image.md5, downloader=downloader)
