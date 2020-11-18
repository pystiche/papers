import os
from argparse import Namespace
from os import path

import torch

import pystiche_papers.sanakoyeu_et_al_2018 as paper
from pystiche.image import write_image
from pystiche_papers import utils


def training(args):
    contents = (
        "garden",
        "bridge_river",
        "glacier_human",
        "mountain",
        "horses",
        "stone_facade",
        "waterway",
        "garden_parc",
    )

    styles = (
        "berthe-morisot",
        "edvard-munch",
        "el-greco",
        "ernst-ludwig-kirchner",
        "jackson-pollock",
        "monet_water-lilies-1914",
        "nicholas-roerich",
        "pablo-picasso",
        "paul-cezanne",
        "samuel-peploe",
        "vincent-van-gogh_road-with-cypresses-1890",
        "wassily-kandinsky",
    )

    images = paper.images()
    images.download(args.image_source_dir)

    content_dataset = paper.content_dataset(
        path.join(args.dataset_dir, "content"), impl_params=args.impl_params
    )

    content_dataset = paper.content_dataset(
        "/home/julianbueltemeier/datasets/places365/data_large_standard",
        impl_params=args.impl_params,
    )

    # content_image_loader = paper.image_loader(
    #     content_dataset,
    #     impl_params=args.impl_params,
    #     pin_memory=str(args.device).startswith("cuda"),
    # )

    for style in styles:
        style_dataset = paper.style_dataset(
            path.join(args.dataset_dir, "style"), style, impl_params=args.impl_params
        )
        style_image_loader = paper.image_loader(
            style_dataset,
            impl_params=args.impl_params,
            pin_memory=str(args.device).startswith("cuda"),
        )

        transformer = paper.training(
            content_image_loader, style_image_loader, impl_params=args.impl_params
        )

        model_name = f"sanakoyeu_et_al_2018__{style}"
        if args.impl_params:
            model_name += "__impl_params"
        utils.save_state_dict(transformer, model_name, root=args.model_dir)

        for content in contents:
            content_image = images[content].read(device=args.device)
            output_image = paper.stylization(
                content_image, transformer, impl_params=args.impl_params,
            )

            output_name = f"{style}_{content}"
            if args.impl_params:
                output_name += "__impl_params"
            output_file = path.join(args.image_results_dir, f"{output_name}.jpg")
            write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    dataset_dir = None
    image_results_dir = None
    model_dir = None
    device = None
    impl_params = True

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "data", "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if dataset_dir is None:
        dataset_dir = path.join(here, "data", "images", "dataset")
    dataset_dir = process_dir(dataset_dir)

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

    return Namespace(
        image_source_dir=image_source_dir,
        dataset_dir=dataset_dir,
        image_results_dir=image_results_dir,
        model_dir=model_dir,
        device=device,
        impl_params=impl_params,
    )


if __name__ == "__main__":
    args = parse_input()
    training(args)
