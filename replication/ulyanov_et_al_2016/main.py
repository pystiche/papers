import os
from argparse import Namespace
from os import path

import torch

import pystiche_papers.ulyanov_et_al_2016 as paper
from pystiche import image, misc, optim
from pystiche_papers import utils


def training_texture(args):
    textures = (
        "cezanne",
        "bricks",
        "pebble",
        "pixels",
        "peppers",
    )

    dataset = paper.dataset(
        path.join(args.dataset_dir, "texture"),
        impl_params=args.impl_params,
        instance_norm=args.instance_norm,
    )
    image_loader = paper.image_loader(
        dataset,
        impl_params=args.impl_params,
        instance_norm=args.instance_norm,
        stylization=args.stylization,
        pin_memory=str(args.device).startswith("cuda"),
    )

    images = paper.images()
    images.download(args.image_source_dir)

    for texture in textures:
        texture_image = images[texture].read(device=args.device)

        transformer = paper.training(
            image_loader,
            texture_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            stylization=args.stylization,
            quiet=args.quiet,
            logger=args.logger,
        )

        model_name = f"ulyanov_et_al_2016__{texture}"
        if args.impl_params:
            model_name += "__impl_params"
        if args.instance_norm:
            model_name += "__instance_norm"
        utils.save_state_dict(transformer, model_name, root=args.model_dir)

        output_image = paper.texture_synthesis(
            torch.empty((1, 1, 256, 256), device="cuda"),
            transformer,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
        )

        output_name = f"{texture}"
        if args.impl_params:
            output_name += "__impl_params"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, "texture", f"{output_name}.jpg")
        image.write_image(output_image, output_file)

        output_image = paper.texture_synthesis(
            torch.empty((1, 1, 512, 512), device="cuda"),
            transformer,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            stylization=args.stylization,
        )

        output_name = f"{texture}"
        output_name += "512_"
        if args.impl_params:
            output_name += "__impl_params"
        if args.instance_norm:
            output_name += "__instance_norm"
        output_file = path.join(args.image_results_dir, "texture", f"{output_name}.jpg")
        image.write_image(output_image, output_file)


def training_style(args):
    contents = (
        "karya",
        "tiger",
        "neckarfront",
        "che_high",
        "the_tower_of_babel" "bird",
        "kitty",
    )
    styles = (
        "candy",
        "the_scream",
        "jean_metzinger",
        "mosaic",
        "pleades",
        "starry",
        "turner",
    )

    dataset = paper.dataset(
        path.join(args.dataset_dir, "style"),
        impl_params=args.impl_params,
        instance_norm=args.instance_norm,
    )
    image_loader = paper.image_loader(
        dataset,
        impl_params=args.impl_params,
        instance_norm=args.instance_norm,
        stylization=args.stylization,
        pin_memory=str(args.device).startswith("cuda"),
    )

    images = paper.images()
    images.download(args.image_source_dir)

    for style in styles:
        style_image = images[style].read(device=args.device)

        transformer = paper.training(
            image_loader,
            style_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            stylization=args.stylization,
            quiet=args.quiet,
            logger=args.logger,
        )

        model_name = f"ulyanov_et_al_2016__{style}"
        if args.impl_params:
            model_name += "__impl_params"
        if args.instance_norm:
            model_name += "__instance_norm"
        utils.save_state_dict(transformer, model_name, root=args.model_dir)

        for content in contents:
            content_image = images[content].read().to(args.device)
            output_image = paper.stylization(
                content_image,
                transformer,
                impl_params=args.impl_params,
                instance_norm=args.instance_norm,
            )

            output_name = f"{style}_{content}"
            if args.impl_params:
                output_name += "__impl_params"
            if args.instance_norm:
                output_name += "__instance_norm"
            output_file = path.join(
                args.image_results_dir, "style", f"{output_name}.jpg"
            )
            image.write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    dataset_dir = None
    model_dir = None
    device = None
    impl_params = True
    instance_norm = False
    stylization = True
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

    device = misc.get_device(device=device)
    logger = optim.OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        device=device,
        impl_params=impl_params,
        instance_norm=instance_norm,
        stylization=stylization,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()
    training_texture(args)
    training_style(args)
