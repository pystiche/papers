import itertools
import os
from argparse import Namespace
from os import path

import pystiche_papers.gatys_ecker_bethge_2015 as paper
from pystiche import image, misc, optim
from pystiche_papers import utils

# FIXME: These values are guessed
NUM_STEPS = 500
SIZE = 500


@utils.abort_if_cuda_memory_exausts
def figure_2(args):
    images = paper.images()
    images.download(args.image_source_dir)

    content_image = images["neckarfront"].read(size=SIZE, device=args.device)

    class StyleImage:
        def __init__(self, label, image, score_weight):
            self.label = label
            self.image = image.read(size=SIZE, device=args.device)
            self.score_weight = score_weight

    style_images = (
        StyleImage("B", images["shipwreck"], 1e3),
        StyleImage("C", images["starry_night"], 1e3),
        StyleImage("D", images["the_scream"], 1e3),
        StyleImage("E", images["femme_nue_assise"], 1e4),
        StyleImage("F", images["composition_vii"], 1e4),
    )

    params = "implementation" if args.impl_params else "paper"
    for style_image in style_images:
        header = f"Replicating Figure 2 {style_image.label} with {params} parameters"
        with args.logger.environment(header):

            style_loss_kwargs = {"score_weight": style_image.score_weight}
            criterion = paper.perceptual_loss(
                impl_params=args.impl_params, style_loss_kwargs=style_loss_kwargs
            )

            output_image = paper.nst(
                content_image,
                style_image.image,
                NUM_STEPS,
                impl_params=args.impl_params,
                criterion=criterion,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(
                args.image_results_dir, f"fig_2__{style_image.label}.jpg"
            )
            args.logger.sep_message(f"Saving result to {output_file}", bottom_sep=False)
            image.write_image(output_image, output_file)


@utils.abort_if_cuda_memory_exausts
def figure_3(args):
    images = paper.images()
    images.download(args.image_source_dir)

    content_image = images["neckarfront"].read(size=SIZE, device=args.device)
    style_image = images["composition_vii"].read(size=SIZE, device=args.device)

    style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    layer_configs = [style_layers[: idx + 1] for idx in range(len(style_layers))]

    score_weights = (1e5, 1e4, 1e3, 1e2)

    for layers, score_weight in itertools.product(layer_configs, score_weights):
        row_label = layers[-1].replace("relu", "Conv")
        column_label = f"{1.0 / score_weight:.0e}"
        header = (
            f"Replicating Figure 3 image in row {row_label} and column {column_label}"
        )
        with args.logger.environment(header):

            style_loss_kwargs = {"layers": layers, "score_weight": score_weight}
            criterion = paper.perceptual_loss(
                impl_params=args.impl_params, style_loss_kwargs=style_loss_kwargs
            )

            output_image = paper.nst(
                content_image,
                style_image,
                NUM_STEPS,
                impl_params=args.impl_params,
                criterion=criterion,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(
                args.image_results_dir, f"fig_3__{row_label}__{column_label}.jpg"
            )
            args.logger.sep_message(f"Saving result to {output_file}", bottom_sep=False)
            image.write_image(output_image, output_file)


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

    device = misc.get_device(device)
    logger = optim.OptimLogger()

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

    figure_2(args)
    figure_3(args)
