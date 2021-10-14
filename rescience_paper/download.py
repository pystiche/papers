import pathlib
import shutil
import tempfile
from typing import Collection

from torchvision.datasets.utils import download_and_extract_archive

HERE = pathlib.Path(__file__).parent


def main():
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    download_template(tmp_dir)
    download_images(tmp_dir)


def download_template(
    tmp_dir: pathlib.Path,
    *,
    commit_hash: str = "75bf717a6b07f0f95e197e7cb8cbdbeee8bb564b",
    excluded: Collection[str] = (
        "bibliography.bib",
        "content.tex",
        "metadata.yaml",
        "README.md",
    ),
) -> None:
    download_and_extract_archive(
        f"https://github.com/ReScience/template/archive/{commit_hash}.zip",
        str(tmp_dir),
    )

    template_dir = tmp_dir / f"template-{commit_hash}"
    for file_or_dir in template_dir.glob("*"):
        if file_or_dir.name not in excluded:
            shutil.move(str(file_or_dir), str(HERE))


def download_images(
    tmp_dir: pathlib.Path,
    *,
    url="https://download.pystiche.org/replication-paper/images.tar.gz",
) -> None:
    download_and_extract_archive(url, str(tmp_dir), extract_root=str(HERE / "graphics"))


if __name__ == "__main__":
    main()
