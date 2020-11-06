import os
from argparse import Namespace
from os import path

import torch

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.image import write_image
from pystiche.optim import OptimLogger


def training_style(args):
    contents = ("TODO",)  # TODO

    styles = ("berthe-morisot",)

    content_dataset = paper.content_dataset()  # TODO
    content_image_loader = paper.image_loader(
        content_dataset,
        impl_params=args.impl_params,
        pin_memory=str(args.device).startswith("cuda"),
    )

    for style in styles:
        style_dataset = paper.style_dataset()  # TODO
        style_image_loader = paper.image_loader(
            style_dataset,
            impl_params=args.impl_params,
            pin_memory=str(args.device).startswith("cuda"),
        )

        transformer = paper.training(
            content_image_loader, style_image_loader, impl_params=args.impl_params
        )

        for content in contents:
            images = paper.images(root=args.image_source_dir)
            content_image = images[content].read().to(args.device)
            output_image = paper.stylization(
                content_image, transformer, impl_params=args.impl_params,
            )

            output_name = f"{style}_{content}"
            if args.impl_params:
                output_name += "__impl_params"
            output_file = path.join(
                args.image_results_dir, "style", f"{output_name}.jpg"
            )
            write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    model_dir = None
    device = None
    impl_params = True
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

    if model_dir is None:
        model_dir = path.join(here, "data", "models")
    model_dir = process_dir(model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    logger = OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        model_dir=model_dir,
        device=device,
        impl_params=impl_params,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    training_style(args)
