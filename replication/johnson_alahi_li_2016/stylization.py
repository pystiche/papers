import os
import re
from os import path

import torch

import pystiche.image.transforms.functional as F
import pystiche_papers.johnson_alahi_li_2016 as paper

from utils import (
    ArgumentParser,
    make_description,
    make_name,
    make_transformer_name,
    read_local_or_builtin_image,
)


def main(args):
    transformer = load_transformer(
        args.models_dir, args.style, args.impl_params, args.instance_norm
    )

    for content in args.content:
        input_image = read_content_image(
            args.images_source_dir,
            content,
            size=1024 if args.instance_norm else 512,
            edge="long",
            device=args.device,
        )

        output_image = paper.stylization(
            input_image,
            transformer,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
        )

        save_ouput_image(
            output_image,
            args.images_results_dir,
            content,
            args.style,
            args.impl_params,
            args.instance_norm,
        )


def load_transformer(model_dir, style, impl_params, instance_norm):
    def load(style=None):
        return paper.transformer(
            style, impl_params=impl_params, instance_norm=instance_norm,
        )

    state_dict = load_local_weights(model_dir, style, impl_params, instance_norm)
    local_weights_available = state_dict is not None
    if local_weights_available:
        transformer = load()
        transformer.load_state_dict(state_dict)
        return transformer
    else:
        try:
            return load(style)
        except RuntimeError:
            print("no local or builtin available for this combination")


def load_local_weights(root, style, impl_params, instance_norm):
    model_name = make_transformer_name(style, impl_params, instance_norm)
    file_pattern = re.compile(model_name + r"-[a-f0-9]{8}[.]pth")

    for file in os.listdir(root):
        match = file_pattern.match(file)
        if match is not None:
            return torch.load(path.join(root, file))
    else:
        return None


def read_content_image(root, content, size=None, edge="short", **read_image_kwargs):
    image = read_local_or_builtin_image(
        root, content, paper.images(), **read_image_kwargs
    )
    return F.resize(image, size, edge=edge)


def save_ouput_image(image, root, content, style, impl_params, instance_norm):
    name = make_name(content, style, impl_params, instance_norm)
    image.write_image(image, path.join(root, f"{name}.jpg"))


def parse_args():
    parser = ArgumentParser(description=make_description("stylization"))

    parser.add_argument("style", type=str, help="Style the transformer was trained on.")
    parser.add_argument(
        "content",
        type=str,
        nargs="*",
        help=(
            "Content images for which the stylization is performed successively. If "
            "relative path, the image is searched in IMAGES_SOURCE_DIR. Can also be a "
            "valid key from the built-in images. Defaults to all built-in content "
            "images."
        ),
    )
    parser.add_images_source_dir_argument()
    parser.add_images_results_dir_argument()
    parser.add_models_dir_argument()
    parser.add_impl_params_and_instance_norm_arguments()
    parser.add_device_argument()

    args = parser.parse_args()
    if not args.content:
        args.content = (
            "chicago",
            "hoovertowernight",
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
