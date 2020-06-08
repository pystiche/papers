import os
from argparse import Namespace
from os import path

from pystiche.image import write_image
from pystiche.misc import get_device
from pystiche.optim import OptimLogger
from pystiche_papers.johnson_alahi_li_2016 import (
    johnson_alahi_li_2016_dataset,
    johnson_alahi_li_2016_image_loader,
    johnson_alahi_li_2016_images,
    johnson_alahi_li_2016_stylization,
    johnson_alahi_li_2016_training,
)
from pystiche_papers.utils import save_state_dict


def training(args):
    content = "chicago"
    styles = (
        "starry_night",
        "la_muse",
        "composition_vii",
        "the_wave",
        "candy",
        "udnie",
        "the_scream",
        "mosaic",
        "feathers",
    )

    dataset = johnson_alahi_li_2016_dataset(
        args.dataset_dir, impl_params=args.impl_params
    )
    image_loader = johnson_alahi_li_2016_image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    images = johnson_alahi_li_2016_images()
    images.download(args.image_source_dir)
    content_image = images[content].read(
        args.image_source_dir,
        size=1024 if args.instance_norm else 512,
        edge="long",
        device=args.device,
    )

    for style in styles:
        style_image = images[style].read(device=args.device)

        transformer = johnson_alahi_li_2016_training(
            image_loader,
            style_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            quiet=args.quiet,
            logger=args.logger,
        )

        model_name = f"johnson_alahi_li_2016__{style}"
        if args.impl_params:
            model_name += "__impl_params"
        if args.instance_norm:
            model_name += "__instance_norm"
        save_state_dict(transformer, model_name, root=args.model_dir)

        output_image = johnson_alahi_li_2016_stylization(
            content_image,
            transformer,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
        )

        output_name = f"{content}__{style}"
        if args.impl_params:
            output_name += "__impl_params"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, f"{output_name}.jpg")
        write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    dataset_dir = None
    model_dir = None
    device = None
    impl_params = True
    instance_norm = False
    quiet = False

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "data", "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "data", "images", "results")
    image_results_dir = process_dir(image_results_dir)

    if dataset_dir is None:
        dataset_dir = path.join(here, "data", "images", "dataset")
    dataset_dir = process_dir(dataset_dir)

    if model_dir is None:
        model_dir = path.join(here, "data", "models")
    model_dir = process_dir(model_dir)

    device = get_device(device)
    logger = OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        device=device,
        impl_params=impl_params,
        instance_norm=instance_norm,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()

    training(args)
