import sys

import pystiche_papers.johnson_alahi_li_2016 as paper
from pystiche import optim
from pystiche_papers.utils import save_state_dict

from utils import (
    ArgumentParser,
    make_description,
    make_transformer_name,
    read_local_or_builtin_image,
)


# The original authors provided models together with the used hyperparameters during
# training. Unfortunately, these hyperparameters not only deviate
# from the paper, but also from the defaults given in the reference implementation.
# Depending on the style and whether instance norm was used or not, the following
# hyperparameters might differ:
# - the score_weight for paper.content_loss(), paper.style_loss(), and
#   paper.regularization(),
# - the size the style_image is resized to with paper.style_transform(), and
# - the number of batches.
def adapted_hyperparameters(impl_params, instance_norm, style):
    hyper_parameters = paper.hyper_parameters()
    if not impl_params:
        return hyper_parameters

    content_loss = hyper_parameters.content_loss
    style_loss = hyper_parameters.style_loss
    regularization = hyper_parameters.regularization
    style_transform = hyper_parameters.style_transform
    batch_sampler = hyper_parameters.batch_sampler

    if instance_norm and style == "candy":
        style_loss.score_weight = 1e1
        regularization.score_weight = 1e-4
        style_transform.edge_size = 384

    elif not instance_norm and style == "composition_vii":
        style_transform.edge_size = 512
        batch_sampler.num_batches = 60000

    elif instance_norm and style == "feathers":
        style_loss.score_weight = 1e1
        regularization.score_weight = 1e-5
        style_transform.edge_size = 180
        batch_sampler.num_batches = 60000

    elif instance_norm and style == "la_muse":
        content_loss.score_weight = 5e-1
        style_loss.score_weight = 1e1
        regularization.score_weight = 1e-4
        style_transform.edge_size = 512

    elif not instance_norm and style == "la_muse":
        regularization.score_weight = 1e-5
        style_transform.edge_size = 512

    elif instance_norm and style == "mosaic":
        style_loss.score_weight = 1e1
        regularization.score_weight = 1e-5
        style_transform.edge_size = 512
        batch_sampler.num_batches = 60000

    elif not instance_norm and style == "starry_night":
        style_loss.score_weight = 3e0
        regularization.score_weight = 1e-5
        style_transform.edge_size = 512

    elif instance_norm and style == "the_scream":
        style_loss.score_weight = 2e1
        regularization.score_weight = 1e-5
        style_transform.edge_size = 384
        batch_sampler.num_batches = 60000

    elif not instance_norm and style == "the_wave":
        regularization.score_weight = 1e-4
        style_transform.edge_size = 512

    elif instance_norm and style == "udnie":
        content_loss.score_weight = 5e-1
        style_loss.score_weight = 1e1

    return hyper_parameters


def main(args):
    dataset = paper.dataset(args.dataset_dir, impl_params=args.impl_params)
    content_image_loader = paper.image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    for style in args.style:
        style_image = read_style_image(
            args.images_source_dir, style, device=args.device
        )

        hyper_parameters = adapted_hyperparameters(
            args.impl_params, args.instance_norm, style
        )

        transformer = paper.training(
            content_image_loader,
            style_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            hyper_parameters=hyper_parameters,
            quiet=args.quiet,
            logger=optim.OptimLogger(),
        )

        model_name = make_transformer_name(style, args.impl_params, args.instance_norm)
        save_state_dict(transformer, model_name, root=args.models_dir)


def read_style_image(root, style, **read_image_kwargs):
    return read_local_or_builtin_image(root, style, paper.images(), **read_image_kwargs)


def make_parser():
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

    return parser


def parse_args(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_parser()
    args = parser.parse_args(args)
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
    args = parse_args()
    main(args)
