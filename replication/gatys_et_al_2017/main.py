import contextlib
import os
from argparse import Namespace
from os import path

import torch

import pystiche_papers.gatys_et_al_2017 as paper
from pystiche.image import write_image
from pystiche.image.transforms.functional import (
    grayscale_to_fakegrayscale,
    resize,
    rgb_to_grayscale,
    rgb_to_yuv,
    transform_channels_affinely,
    yuv_to_rgb,
)
from pystiche.misc import get_device
from pystiche.optim import OptimLogger
from pystiche_papers.utils import abort_if_cuda_memory_exausts


@contextlib.contextmanager
def replicate_figure(logger, figure, impl_params):
    params = "implementation" if impl_params else "paper"
    header = f"Replicating {figure} with {params} parameters"
    with logger.environment(header):
        yield


def log_saving_info(logger, output_file):
    logger.sep_message(f"Saving result to {output_file}", bottom_sep=False)


def read_image_and_guides(image, **read_kwargs):
    return image.read(**read_kwargs), image.guides.read(**read_kwargs)


def figure_2(args):
    @abort_if_cuda_memory_exausts
    def figure_2_d(content_image, style_image):
        with replicate_figure(args.logger, "2 (d)", args.impl_params):
            output_image = paper.nst(
                content_image,
                style_image,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(args.image_results_dir, "fig_2__d.jpg")
            log_saving_info(args.logger, output_file)
            write_image(output_image, output_file)

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
        with replicate_figure(args.logger, f"2 ({label})", args.impl_params):

            output_image = paper.guided_nst(
                content_image,
                content_guides,
                style_images_and_guides,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(args.image_results_dir, f"fig_2__{label}.jpg")
            log_saving_info(args.logger, output_file)
            write_image(output_image, output_file)

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
                torch.cholesky(target_cov), torch.inverse(torch.cholesky(input_cov))
            )
        else:
            # FIXME: add error message
            raise ValueError
        matrix = matrix.unsqueeze(0)

        bias = target_mean - torch.bmm(matrix, input_mean)

        return transform_channels_affinely(input, matrix, bias)

    @abort_if_cuda_memory_exausts
    def figure_3_c(content_image, style_image):
        with replicate_figure(args.logger, "3 (c)", args.impl_params):
            output_image = paper.nst(
                content_image,
                style_image,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(args.image_results_dir, "fig_3__c.jpg")
            log_saving_info(args.logger, output_file)
            write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_d(content_image, style_image):
        content_image_yuv = rgb_to_yuv(content_image)
        content_luminance = grayscale_to_fakegrayscale(content_image_yuv[:, :1])
        content_chromaticity = content_image_yuv[:, 1:]

        style_luminance = grayscale_to_fakegrayscale(rgb_to_grayscale(style_image))

        with replicate_figure(args.logger, "3 (d)", args.impl_params):
            output_luminance = paper.nst(
                content_luminance,
                style_luminance,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )
            output_luminance = torch.mean(output_luminance, dim=1, keepdim=True)
            output_chromaticity = resize(
                content_chromaticity, output_luminance.size()[2:]
            )
            output_image_yuv = torch.cat((output_luminance, output_chromaticity), dim=1)
            output_image = yuv_to_rgb(output_image_yuv)

            output_file = path.join(args.image_results_dir, "fig_3__d.jpg")
            log_saving_info(args.logger, output_file)
            write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_e(content_image, style_image, method="cholesky"):
        style_image = match_channelwise_statistics(style_image, content_image, method)

        with replicate_figure(args.logger, "3 (e)", args.impl_params):
            output_image = paper.nst(
                content_image,
                style_image,
                impl_params=args.impl_params,
                quiet=args.quiet,
                logger=args.logger,
            )

            output_file = path.join(args.image_results_dir, "fig_3__e.jpg")
            log_saving_info(args.logger, output_file)
            write_image(output_image, output_file)

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


def parse_input():
    # TODO: write CLI
    image_source_dir = None
    image_guides_dir = None
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

    if image_guides_dir is None:
        image_guides_dir = path.join(here, "images", "guides")
    image_guides_dir = process_dir(image_guides_dir)

    if image_results_dir is None:
        image_results_dir = path.join(here, "images", "results")
    image_results_dir = process_dir(image_results_dir)

    device = get_device()
    logger = OptimLogger()

    return Namespace(
        image_source_dir=image_source_dir,
        image_guides_dir=image_guides_dir,
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
