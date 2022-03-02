import os
from typing import Sequence,  Tuple, cast
from argparse import Namespace
from os import path

import torch
from torch import nn
from pystiche import misc, optim, loss, enc, ops, image, data


def compute_layer_weights(
    layers: Sequence[str], multi_layer_encoder: enc.MultiLayerEncoder
) -> Tuple[float, ...]:
    def find_out_channels(multi_layer_encoder: nn.Module, layer: str) -> int:
        modules = multi_layer_encoder._modules
        layers = list(modules.keys())
        layers = reversed(layers[: layers.index(layer) + 1])
        for layer_ in layers:
            try:
                return cast(int, modules[layer_].out_channels)
            except AttributeError:
                pass

        raise RuntimeError(
            f"Neither '{layer}' nor any previous layer has an 'out_channels' "
            f"attribute."
        )


    num_channels = []
    for layer in layers:
        if layer not in multi_layer_encoder:
            raise ValueError(f"Layer {layer} is not part of the multi_layer_encoder.")

        num_channels.append(find_out_channels(multi_layer_encoder, layer))

    return tuple(1.0 / n ** 2.0 for n in num_channels)


def nst_images():
    images_ = {
        "bird": data.DownloadableImage(
            "https://free-images.com/md/b9c1/colorful_bird_rainbow_parakeet.jpg",
            license=data.PublicDomainLicense(),
            md5="3eab0e8a32e020b40536154acdc05ab4",
            file="colorful_bird.jpg",
        ),
        "dog": data.DownloadableImage(
            "https://free-images.com/lg/4444/dog_animal_puppy_animals_1.jpg",
            license=data.PublicDomainLicense(),
            md5="2312d5003f5f96fa72d9c61dd37cd411",
            file="dog_animal.jpg",
        ),
        "duck": data.DownloadableImage(
            "https://free-images.com/md/79c0/duck_ducks_animal_pets_0.jpg",
            license=data.PublicDomainLicense(),
            md5="10609bb04100e9e1172d7c2202f1c05d",
            file="duck.jpg",
        ),
        "bird2": data.DownloadableImage(
            "https://free-images.com/lg/0902/parrot_bird_colorful_plumage.jpg",
            license=data.PublicDomainLicense(),
            md5="40e9c220154ae7462c70a1cedbd86bd4",
            file="parrot.jpg",
        ),
        "bird3": data.DownloadableImage(
            "https://free-images.com/md/4c5a/parrot_bird_ara_colorful_2.jpg",
            license=data.PublicDomainLicense(),
            md5="4a42c7ce24619b2c7b32526b3b42dea3",
            file="parrot_bird.jpg",
        ),
        "turtle": data.DownloadableImage(
            "https://free-images.com/md/1d4f/turtle_green_turtle_ocean.jpg",
            license=data.PublicDomainLicense(),
            md5="064742c2f648240c5d28fe1c421a8a90",
            file="turtle_green.jpg",
        ),
        "mosaic": data.DownloadableImage(
            "https://free-images.com/md/ab85/mosaic_stones_structure_pattern.jpg",
            license=data.PublicDomainLicense(),
            md5="afa9e5024aff029753a6901cdc19bedc",
            file="mosaic_stones.jpg",
        ),
        "stones": data.DownloadableImage(
            "https://free-images.com/md/b156/stone_floor_pebbles_stone_1.jpg",
            license=data.PublicDomainLicense(),
            md5="420beb77451956828cf1bf00d6a4061f",
            file="stone_floor.jpg",
        ),
        "abstract": data.DownloadableImage(
            "https://free-images.com/md/b2ce/light_creative_abstract_colorful.jpg",
            license=data.PublicDomainLicense(),
            md5="ff45182232f05ee2eb63ac8881173609",
            file="light_creative.jpg",
        ),
        "starry_night": data.DownloadableImage(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            title="Starry Night",
            author="Vincent van Gogh",
            date="ca. 1889",
            license=data.ExpiredCopyrightLicense(1890),
            md5="372e5bc438e3e8d0eb52cc6f7ef44760",
        ),
    }
    return data.DownloadableImageCollection(images_)


def nst(
        args,
        content_image,
        style_image,
        output_name,
        content_layer="relu4_2",
        content_weight=1e0,
        style_layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1"),
        style_weight=1e3,
        starting_point="content",
        num_steps=500,
):
    multi_layer_encoder = enc.vgg19_multi_layer_encoder(
        weights="caffe",
        internal_preprocessing=True,
        allow_inplace=True
    )

    # content loss
    content_encoder = multi_layer_encoder.extract_encoder(content_layer)
    content_loss = ops.FeatureReconstructionOperator(
        content_encoder,
        score_weight=content_weight
    )

    # style loss
    layer_weights = compute_layer_weights(style_layers, multi_layer_encoder)
    def get_style_op(encoder, layer_weight):
        return ops.GramOperator(encoder, score_weight=layer_weight)
    style_loss = ops.MultiLayerEncodingOperator(
        multi_layer_encoder,
        style_layers,
        get_style_op,
        score_weight=style_weight,
        layer_weights=layer_weights
    )

    perceptual_loss = loss.PerceptualLoss(content_loss, style_loss).to(args.device)

    images = nst_images()
    images.download(args.image_source_dir)
    perceptual_loss.set_content_image(content_image)
    perceptual_loss.set_style_image(style_image)
    input_image = misc.get_input_image(starting_point, content_image=content_image)

    output_image = optim.image_optimization(input_image, perceptual_loss, num_steps=num_steps)

    output_file = path.join(args.image_results_dir, output_name)
    image.write_image(output_image, output_file)


def full_nst(args, content, style, size=500):
    contents = [content,] if content is not None else ["dog", "dog", "duck", "bird2", "bird3", "turtle"]
    styles = [style,] if style is not None else ["starry_night", "abstract", "stones",]

    images = nst_images()
    images.download(args.image_source_dir)
    for content in contents:
        for style in styles:
            content_image = images[content].read(
                size=size, device=args.device
            )
            style_image = images[style].read(
                size=size, device=args.device
            )
            name = f"nst_IST_paper_{content}__{style}__full.jpg"
            nst(args, content_image, style_image, name)


def layer_nst(args, content, style, size=500):
    images = nst_images()
    images.download(args.image_source_dir)

    content_image = images[content].read(
        size=size, device=args.device
    )
    style_image = images[style].read(
        size=size, device=args.device
    )

    # style loss
    style_layers = ["relu1_1", "relu2_2", "relu3_1", "relu4_1",]

    for layer in style_layers:
        name = f"nst_IST_paper_{content}__{style}__{layer}.jpg"
        nst(args, content_image, style_image, name, style_layers=[layer,], style_weight=1e1, starting_point="random")


def init_nst(args, content, style, size=500):
    images = nst_images()
    images.download(args.image_source_dir)

    content_image = images[content].read(
        size=size, device=args.device
    )
    style_image = images[style].read(
        size=size, device=args.device
    )

    for starting_point in ["random", "content"]:
        name = f"nst_IST_paper_{content}__{style}__start_{starting_point}.jpg"
        nst(args, content_image, style_image, name, style_weight=1e1, num_steps=1000, starting_point=starting_point)


def iteration_nst(args, content, style, size=500):
    images = nst_images()
    images.download(args.image_source_dir)

    content_image = images[content].read(
        size=size, device=args.device
    )
    style_image = images[style].read(
        size=size, device=args.device
    )

    for num_steps in [100, 250, 500]:
        name = f"nst_IST_paper_{content}__{style}__iteration_{num_steps}.jpg"
        nst(args, content_image, style_image, name, num_steps=num_steps, starting_point="random", style_weight=1e1)


def image_size_nst(args, content, style):
    images = nst_images()
    images.download(args.image_source_dir)
    for image_size in [256, 512]:
        content_image = images[content].read(
            size=image_size, device=args.device
        )
        style_image = images[style].read(
            size=image_size, device=args.device
        )

        name = f"nst_IST_paper_{content}__{style}__size_{image_size}.jpg"
        nst(args, content_image, style_image, name)


def weights_nst(args, content, style, size=500):
    images = nst_images()
    images.download(args.image_source_dir)
    for weight in [1e0, 1e1, 1e2, 1e3]:
        content_image = images[content].read(
            size=size, device=args.device
        )
        style_image = images[style].read(
            size=size, device=args.device
        )

        name = f"nst_IST_paper_{content}__{style}__style_weight_{weight}.jpg"
        nst(args, content_image, style_image, name, style_weight=weight, starting_point="random", num_steps=1000)


def style_generation_nst(args, size=500):
    style = "starry_night"
    images = nst_images()
    images.download(args.image_source_dir)
    content_image = images[style].read(
        size=size, device=args.device
    )
    style_image = images[style].read(
        size=size, device=args.device
    )

    name = f"nst_IST_paper__{style}__generated.jpg"
    nst(args, content_image, style_image, name, content_weight=0, starting_point="random", num_steps=1000)


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_results_dir = None
    device = None

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "graphics", "images", "nst", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "graphics", "images", "nst", "results")
    image_results_dir = process_dir(image_results_dir)

    device = misc.get_device(device=device)
    logger = optim.OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_results_dir=image_results_dir,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    args = parse_input()
    content = "bird"
    style = "mosaic"
    full_nst(args, content, style)
    layer_nst(args, content, style)
    init_nst(args, content, style)
    iteration_nst(args, content, style)
    image_size_nst(args, content, style)
    weights_nst(args, content, style)
    style_generation_nst(args)
