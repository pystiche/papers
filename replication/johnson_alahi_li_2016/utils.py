import argparse
import os
from os import path

from pystiche.image import read_image
from pystiche.misc import get_device

__all__ = [
    "make_description",
    "make_name",
    "make_transformer_name",
    "read_local_or_builtin_image",
    "ArgumentParser",
]


def make_description(obj):
    return (
        f"{obj.capitalize()} for the paper 'Perceptual losses for real-time style "
        "transfer and super-resolution' by Johnson, Alahi, and Li."
    )


def make_name(prefix, style, impl_params, instance_norm):
    name_parts = [
        prefix,
        path.splitext(path.basename(style))[0],
    ]
    if impl_params:
        name_parts.append("impl_params")
    if instance_norm:
        name_parts.append("instance_norm")
    return "__".join(name_parts)


def make_transformer_name(style, impl_params, instance_norm):
    return make_name(
        "johnson_alahi_li_2016_transformer", style, impl_params, instance_norm
    )


def read_local_or_builtin_image(root, name, builtin_images, **read_image_kwargs):
    file = name
    if not path.abspath(file):
        file = path.join(root, name)
    if path.exists(file):
        return read_image(file, **read_image_kwargs)

    return builtin_images[name].read(root, **read_image_kwargs)


class ArgumentParser(argparse.ArgumentParser):
    def add_data_dir_argument(self, data_dir, *name_or_flags):
        rel = path.join("data", data_dir)
        abs = path.join(path.dirname(__file__), rel)
        parts = data_dir.split(os.sep)

        if not name_or_flags:
            name_or_flags = (f"--{'-'.join(parts + ['dir'])}",)
        type = str
        default = abs
        help = (
            f"{' '.join((parts[0].capitalize(), *parts[1:]))} directory. Defaults to "
            f"{rel} relative to this script."
        )

        self.add_argument(*name_or_flags, type=type, default=default, help=help)

    def add_images_source_dir_argument(self):
        self.add_data_dir_argument(path.join("images", "source"),)

    def add_images_results_dir_argument(self):
        self.add_data_dir_argument(path.join("images", "results"))

    def add_dataset_dir_argument(self):
        self.add_data_dir_argument("dataset")

    def add_models_dir_argument(self):
        self.add_data_dir_argument("models")

    def _process_data_dirs(self, args):
        for arg, val in vars(args).items():
            if not arg.endswith("dir"):
                continue

            dir = path.abspath(path.expanduser(val))
            os.makedirs(dir, exist_ok=True)
            setattr(args, arg, dir)

    def add_impl_params_and_instance_norm_arguments(self):
        self.add_argument(
            "--no-impl-params",
            action="store_true",
            default=False,
            help=(
                "If given, use the parameters reported in the paper rather than the "
                "ones used in the implementation."
            ),
        )
        self.add_argument(
            "--no-instance-norm",
            action="store_true",
            default=False,
            help="If given, use batch rather than instance normalization.",
        )

    def _process_impl_params_and_instance_norm(self, args):
        args.impl_params = not args.no_impl_params
        del args.no_impl_params

        args.instance_norm = not args.no_instance_norm
        del args.no_instance_norm

    def add_device_argument(self):
        self.add_argument(
            "--device",
            type=str,
            default=None,
            help=(
                "Device the training is performed on. Defaults to the available "
                "hardware preferring CUDA over CPU."
            ),
        )

    def _process_device(self, args):
        if "device" in args:
            args.device = get_device(args.device)

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args=args, namespace=namespace)
        self._process_data_dirs(args)
        self._process_impl_params_and_instance_norm(args)
        self._process_device(args)
        return args
