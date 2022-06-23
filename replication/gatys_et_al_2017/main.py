import os
from argparse import Namespace
from os import path

import torch
from torchvision.transforms.functional import resize, rgb_to_grayscale

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche.image import write_image
from pystiche.misc import get_device
from pystiche_papers.utils import abort_if_cuda_memory_exausts, make_output_filename


def read_image_and_guides(image, **read_kwargs):
    return image.read(**read_kwargs), image.guides.read(**read_kwargs)


def save_result(output_image, output_file):
    print(f"Saving result to {output_file}")
    write_image(output_image, output_file)
    print("#" * int(os.environ.get("COLUMNS", "80")))


def figure_2(args):
    @abort_if_cuda_memory_exausts
    def figure_2_d(content_image, style_image):
        print("Replicating Figure 2 (d)")
        output_image = paper.nst(
            content_image,
            style_image,
            impl_params=args.impl_params,
        )
        filename = make_output_filename(
            "gatys_et_al_2017", 
            "fig_2", 
            "d", 
            impl_params=args.impl_params,
        )

        output_file = path.join(args.image_results_dir, filename)
        save_result(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_2_ef(
        label,
        content_image,
        content_building_guide,
        content_sky_guide,
        style_building_image,
        style_building_guide,
        style_sky_image,
        style_sky_guide,
    ):
        content_guides = {"building": content_building_guide, "sky": content_sky_guide}
        style_images_and_guides = {
            "building": (style_building_image, style_building_guide),
            "sky": (style_sky_image, style_sky_guide),
        }

        print(f"Replicating Figure 2 ({label})")
        output_image = paper.guided_nst(
            content_image,
            content_guides,
            style_images_and_guides,
            impl_params=args.impl_params,
        )
        filename = make_output_filename(
            "gatys_et_al_2017", "fig_2", label, impl_params=args.impl_params
        )
        output_file = path.join(args.image_results_dir, filename)
        save_result(output_image, output_file)

    images = paper.images()

    content_image, content_guides = read_image_and_guides(
        images["house"], root=args.image_source_dir, device=args.device
    )

    style1_image, style1_guides = read_image_and_guides(
        images["watertown"], root=args.image_source_dir, device=args.device
    )

    style2_image, style2_guides = read_image_and_guides(
        images["wheat_field"], root=args.image_source_dir, device=args.device
    )

    figure_2_d(content_image, style1_image)

    figure_2_ef(
        "e",
        content_image,
        content_guides["building"],
        content_guides["sky"],
        style1_image,
        style1_guides["building"],
        style1_image,
        style1_guides["sky"],
    )

    figure_2_ef(
        "f",
        content_image,
        content_guides["building"],
        content_guides["sky"],
        style1_image,
        style1_guides["building"],
        style2_image,
        style2_guides["sky"],
    )


def figure_3(args):
    def calculate_channelwise_mean_covariance(image):
        batch_size, num_channels, height, width = image.size()
        num_pixels = height * width
        image = image.view(batch_size, num_channels, num_pixels)

        mean = torch.mean(image, dim=2, keepdim=True)

        image_centered = image - mean
        cov = torch.bmm(image_centered, image_centered.transpose(1, 2)) / num_pixels

        return mean, cov

    def msqrt(x):
        e, v = torch.symeig(x, eigenvectors=True)
        return torch.chain_matmul(v, torch.diag(e), v.t())

    def match_channelwise_statistics(input, target, method):
        input_mean, input_cov = calculate_channelwise_mean_covariance(input)
        target_mean, target_cov = calculate_channelwise_mean_covariance(target)

        input_cov, target_cov = [cov.squeeze(0) for cov in (input_cov, target_cov)]
        if method == "image_analogies":
            matrix = torch.mm(msqrt(target_cov), torch.inverse(msqrt(input_cov)))
        elif method == "cholesky":
            matrix = torch.mm(
                torch.linalg.cholesky(target_cov),
                torch.inverse(torch.linalg.cholesky(input_cov)),
            )
        else:
            # FIXME: add error message
            raise ValueError
        matrix = matrix.unsqueeze(0)

        bias = target_mean - torch.bmm(matrix, input_mean)

        return transform_channels_affinely(input, matrix, bias)

    @abort_if_cuda_memory_exausts
    def figure_3_c(content_image, style_image):
        print("Replicating Figure 3 (c)")
        output_image = paper.nst(
            content_image,
            style_image,
            impl_params=args.impl_params,
        )
        filename = make_output_filename(
            "gatys_et_al_2017", "fig_3", "c", impl_params=args.impl_params
        )
        output_file = path.join(args.image_results_dir, filename)
        save_result(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_d(content_image, style_image):
        content_image_yuv = rgb_to_yuv(content_image)
        content_luminance = content_image_yuv[:, :1].repeat(1, 3, 1, 1)
        content_chromaticity = content_image_yuv[:, 1:]

        style_luminance = rgb_to_grayscale(style_image, num_output_channels=3)

        print("Replicating Figure 3 (d)")
        output_luminance = paper.nst(
            content_luminance,
            style_luminance,
            impl_params=args.impl_params,
        )
        output_luminance = torch.mean(output_luminance, dim=1, keepdim=True)
        output_chromaticity = resize(content_chromaticity, output_luminance.size()[2:])
        output_image_yuv = torch.cat((output_luminance, output_chromaticity), dim=1)
        output_image = yuv_to_rgb(output_image_yuv)
        filename = make_output_filename(
            "gatys_et_al_2017", "fig_3", "d", impl_params=args.impl_params
        )
        output_file = path.join(args.image_results_dir, filename)
        save_result(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_e(content_image, style_image, method="cholesky"):
        style_image = match_channelwise_statistics(style_image, content_image, method)

        print("Replicating Figure 3 (e)")
        output_image = paper.nst(
            content_image,
            style_image,
            impl_params=args.impl_params,
        )
        filename = make_output_filename(
            "gatys_et_al_2017", "fig_3", "e", impl_params=args.impl_params
        )
        output_file = path.join(args.image_results_dir, filename)
        save_result(output_image, output_file)

    images = paper.images()
    content_image = images["schultenhof"].read(
        root=args.image_source_dir, device=args.device
    )
    style_image = images["starry_night"].read(
        root=args.image_source_dir, device=args.device
    )

    figure_3_c(content_image, style_image)
    figure_3_d(content_image, style_image)
    figure_3_e(content_image, style_image)


def transform_channels_affinely(x: torch.Tensor, matrix, bias=None):
    batch_size, _, *spatial_size = x.size()
    x = torch.flatten(x, 2)

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    num_channels = matrix.size()[1]

    x = torch.bmm(matrix, x)

    if bias is not None:
        if bias.dim() == 2:
            bias = bias.unsqueeze(0)
        x += bias

    return x.view(batch_size, num_channels, *spatial_size)


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        (
            (0.299, 0.587, 0.114),
            (-0.147, -0.289, 0.436),
            (0.615, -0.515, -0.100),
        ),
    )
    return transform_channels_affinely(x, transformation_matrix.to(x))


def yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    transformation_matrix = torch.tensor(
        (
            (1.000, 0.000, 1.140),
            (1.000, -0.395, -0.581),
            (1.000, 2.032, 0.000),
        ),
    )
    return transform_channels_affinely(x, transformation_matrix.to(x))


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_guides_dir = None
    image_results_dir = None
    impl_params = True

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_source_dir is None:
        image_source_dir = path.join(here, "images", "source")
    image_source_dir = process_dir(image_source_dir)

    if image_guides_dir is None:
        image_guides_dir = path.join(here, "images", "guides")
    image_guides_dir = process_dir(image_guides_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "images", "results")
    image_results_dir = process_dir(image_results_dir)

    device = get_device()

    return Namespace(
        image_source_dir=image_source_dir,
        image_guides_dir=image_guides_dir,
        image_results_dir=image_results_dir,
        device=device,
        impl_params=impl_params,
    )


if __name__ == "__main__":
    args = parse_input()

    figure_2(args)
    figure_3(args)
