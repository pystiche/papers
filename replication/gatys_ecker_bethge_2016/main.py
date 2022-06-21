import itertools
import os
from argparse import Namespace
from os import path

import pystiche_papers.gatys_ecker_bethge_2016 as paper
from pystiche import image, misc
from pystiche_papers import utils


def save_result(output_image, output_file):
    print(f"Saving result to {output_file}")
    image.write_image(output_image, output_file)
    print("#" * int(os.environ.get("COLUMNS", "80")))


@utils.abort_if_cuda_memory_exausts
def figure_2(args):
    images = paper.images()
    images.download(args.image_source_dir)

    hyper_parameters = paper.hyper_parameters()

    content_image = images["neckarfront"].read(
        size=hyper_parameters.nst.image_size, device=args.device
    )

    class StyleImage:
        def __init__(self, label, image, score_weight):
            self.label = label
            self.image = image.read(
                size=hyper_parameters.nst.image_size, device=args.device
            )
            self.score_weight = score_weight

    style_images = (
        StyleImage("B", images["shipwreck"], 1e3),
        StyleImage("C", images["starry_night"], 1e3),
        StyleImage("D", images["the_scream"], 1e3),
        StyleImage("E", images["femme_nue_assise"], 1e4),
        StyleImage("F", images["composition_vii"], 1e4),
    )

    for style_image in style_images:
        print(f"Replicating Figure 2 {style_image.label}")
        hyper_parameters.style_loss.score_weight = style_image.score_weight

        output_image = paper.nst(
            content_image,
            style_image.image,
            impl_params=args.impl_params,
            hyper_parameters=hyper_parameters,
        )
        filename = utils.make_output_filename(
            "gatys_ecker_betghe_2016", "fig_2", style_image.label,
            impl_params=args.impl_params,
        )
        save_result(
            output_image,
            path.join(args.image_results_dir, filename),
        )


@utils.abort_if_cuda_memory_exausts
def figure_3(args):
    images = paper.images()
    images.download(args.image_source_dir)

    hyper_parameters = paper.hyper_parameters(args.impl_params)

    content_image = images["neckarfront"].read(
        size=hyper_parameters.nst.image_size, device=args.device
    )
    style_image = images["composition_vii"].read(
        size=hyper_parameters.nst.image_size, device=args.device
    )

    style_layers = hyper_parameters.style_loss.layers
    layer_configs = [style_layers[: idx + 1] for idx in range(len(style_layers))]

    score_weights = (1e5, 1e4, 1e3, 1e2)

    params = "implementation" if args.impl_params else "paper"
    for layers, score_weight in itertools.product(layer_configs, score_weights):
        row_label = layers[-1]
        column_label = f"{1.0 / score_weight:.0e}"
        print(
            f"Replicating Figure 3 image in row {row_label} and column {column_label} "
            f"with {params} parameters."
        )
        hyper_parameters.style_loss.layers = layers
        if args.impl_params:
            hyper_parameters.style_loss.layer_weights = paper.compute_layer_weights(
                layers
            )
        hyper_parameters.style_loss.score_weight = score_weight

        output_image = paper.nst(
            content_image,
            style_image,
            impl_params=args.impl_params,
            hyper_parameters=hyper_parameters,
        )

        filename = utils.make_output_filename(
            ["gatys_ecker_betghe_2016", "fig_3", row_label, column_label],
            impl_params=args.impl_params,
        )
        save_result(
            output_image,
            path.join(args.image_results_dir, filename),
        )


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

    device = misc.get_device(device)

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        impl_params=impl_params,
    )


if __name__ == "__main__":
    args = parse_input()

    figure_2(args)
    figure_3(args)
