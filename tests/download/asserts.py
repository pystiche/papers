from os import path
from time import sleep
from urllib.request import Request, urlopen

from torchvision.datasets.utils import calculate_md5

from tests.utils import get_tempdir

__all__ = [
    "assert_is_downloadable",
    "assert_downloads_correctly",
    "assert_image_is_downloadable",
    "assert_image_downloads_correctly",
]

USER_AGENT = "pystiche_papers/test_suite"


def retry(fn, times=1, wait=5.0):
    if not times:
        return fn()

    msgs = []
    for _ in range(times + 1):
        try:
            return fn()
        except AssertionError as error:
            msgs.append(str(error))
            sleep(wait)
    else:
        head = (
            f"Assertion failed {times + 1} times with {wait:.1f} seconds intermediate "
            f"wait time.\n"
        )
        raise AssertionError(
            "\n".join((head, *(f"{idx}: {error}" for idx, error in enumerate(msgs, 1))))
        )


def assert_response_ok(response, url=None):
    msg = f"The server returned status code {response.code}"
    if url is not None:
        msg += f" for the URL {url}"
    assert response.code == 200, msg


def assert_is_downloadable(url, times=1, wait=5.0):
    response = urlopen(Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD"))
    retry(lambda: assert_response_ok(response, url), times=times - 1, wait=wait)


def default_downloader(url, root):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    file = path.join(root, path.basename(url))
    with urlopen(request) as response, open(file, "wb") as fh:
        assert_response_ok(response, url)
        fh.write(response.read())
    return file


def assert_downloads_correctly(
    url, md5, downloader=default_downloader, times=1, wait=5.0
):
    with get_tempdir() as root:
        file = retry(lambda: downloader(url, root), times=times - 1, wait=wait)
        assert calculate_md5(file) == md5, "The MD5 checksums mismatch"


def assert_image_is_downloadable(image, **kwargs):
    assert_is_downloadable(image.url, **kwargs)


def assert_image_downloads_correctly(image, **kwargs):
    def downloader(url, root):
        image.download(root=root)
        return path.join(root, image.file)

    assert_downloads_correctly(None, image.md5, downloader=downloader, **kwargs)
