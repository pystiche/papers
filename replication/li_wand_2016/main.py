import os
from argparse import Namespace
from os import path

import pystiche_papers.li_wand_2016 as paper
from pystiche.image import write_image
from pystiche.misc import get_device
from pystiche_papers.utils import abort_if_cuda_memory_exausts


@abort_if_cuda_memory_exausts
def figure_6(args):
    images = paper.images()
    images.download(args.image_source_dir)
    positions = ("top", "bottom")
    image_pairs = (
        (images["blue_bottle"], images["self-portrait"]),
        (images["s"], images["composition_viii"]),
    )

    for position, image_pair in zip(positions, image_pairs):
        content_image = image_pair[0].read(device=args.device)
        style_image = image_pair[1].read(device=args.device)

        print(
            f"Replicating the {position} half of figure 6 "
            f"with {'implementation' if args.impl_params else 'paper'} parameters"
        )

        hyper_parameters = paper.hyper_parameters(impl_params=args.impl_params)
        if args.impl_params:
            # https://github.com/pmeier/CNNMRF/blob/fddcf4d01e2a6ce201059d8bc38597f74a09ba3f/run_trans.lua#L66
            hyper_parameters.content_loss.layer = "relu4_2"
            hyper_parameters.target_transforms.num_scale_steps = 1
            hyper_parameters.target_transforms.num_rotate_steps = 1

        output_image = paper.nst(
            content_image,
            style_image,
            impl_params=args.impl_params,
            hyper_parameters=hyper_parameters,
        )

        output_file = path.join(args.image_results_dir, f"fig_6__{position}.jpg")
        print(f"Saving result to {output_file}")
        write_image(output_image, output_file)
        print("#" * int(os.environ.get("COLUMNS", "80")))


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    device = None
    impl_params = True

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

    device = get_device(device)

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        impl_params=impl_params,
    )


if __name__ == "__main__":
    args = parse_input()

    figure_6(args)
