import os
from argparse import Namespace
from os import path

import torch

from pystiche.image import write_image
from pystiche.optim import OptimLogger
from pystiche_papers.li_wand_2016 import li_wand_2016_images, li_wand_2016_nst
from pystiche_papers.utils import abort_if_cuda_memory_exausts


@abort_if_cuda_memory_exausts
def figure_6(args):
    images = li_wand_2016_images()
    images.download(root=args.image_source_dir)
    positions = ("top", "bottom")
    image_pairs = (
        (images["blue_bottle"], images["self-portrait"]),
        (images["s"], images["composition_viii"]),
    )

    for position, image_pair in zip(positions, image_pairs):
        content_image = image_pair[0].read(device=args.device)
        style_image = image_pair[1].read(device=args.device)

        params = "implementation" if args.impl_params else "paper"
        header = (
            f"Replicating the {position} half of figure 6 " f"with {params} parameters"
        )
        with args.logger.environment(header):
            output_image = li_wand_2016_nst(
                content_image,
                style_image,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(args.image_results_dir, f"fig_6__{position}.jpg")
            args.logger.sep_message(f"Saving result to {output_file}", bottom_sep=False)
            write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    device = None
    impl_params = True
    quiet = False

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "images", "results")
    image_results_dir = process_dir(image_results_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    logger = OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        impl_params=impl_params,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()

    figure_6(args)
