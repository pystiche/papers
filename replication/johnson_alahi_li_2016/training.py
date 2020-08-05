import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import optim
from pystiche_papers.utils import save_state_dict

from utils import (
    ArgumentParser,
    make_description,
    make_transformer_name,
    read_local_or_builtin_image,
)


def main(args):
    dataset = paper.dataset(args.dataset_dir, impl_params=args.impl_params)
    content_image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    for style in args.style:
        style_image = read_style_image(
            args.images_source_dir, style, device=args.device
        )

        transformer = paper.training(
            content_image_loader,
            style_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            quiet=args.quiet,
            logger=optim.OptimLogger(),
        )

        model_name = make_transformer_name(style, args.impl_params, args.instance_norm)
        save_state_dict(transformer, model_name, root=args.model_dir)


def read_style_image(root, style, **read_image_kwargs):
    return read_local_or_builtin_image(root, style, paper.images(), **read_image_kwargs)


def parse_input():
    parser = ArgumentParser(description=make_description("training"))

    parser.add_argument(
        "style",
        type=str,
        nargs="*",
        help=(
            "Style images for which the training is performed successively. If "
            "relative path, the image is searched in IMAGES_SOURCE_DIR. Can also be a "
            "valid key from the built-in images. Defaults to all built-in style images."
        ),
    )
    parser.add_images_source_dir_argument()
    parser.add_models_dir_argument()
    parser.add_dataset_dir_argument()
    parser.add_impl_params_and_instance_norm_arguments()
    parser.add_device_argument()
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print training information to STDOUT.",
    )

    args = parser.parse_args()
    if not args.style:
        args.style = (
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
    return args


if __name__ == "__main__":
    args = parse_input()
    main(args)
