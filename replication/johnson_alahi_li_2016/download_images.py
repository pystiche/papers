from torchvision.datasets.utils import download_and_extract_archive

from pystiche_papers import johnson_alahi_li_2016 as paper
from utils import ArgumentParser, make_description


def main(args):
    paper.johnson_alahi_li_2016_images().download(args.images_source_dir)

    if args.no_dataset:
        return

    download_and_extract_archive(
        "http://images.cocodataset.org/zips/train2014.zip",
        args.dataset_dir,
        md5="0da8c0bd3d6becc4dcb32757491aca88",
    )


def parse_args():
    parser = ArgumentParser(description=make_description("download the images"))

    parser.add_images_source_dir_argument()
    parser.add_dataset_dir_argument()
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="If given, do not download the dataset (~13GB).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
