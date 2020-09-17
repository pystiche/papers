import os
from collections import OrderedDict
from copy import copy
from os import path

import tensorflow as tf

import torch
from torchvision.datasets.utils import download_and_extract_archive

from pystiche_papers.utils import get_tmp_dir, save_state_dict

URL_FMTSTR = "https://hcicloud.iwr.uni-heidelberg.de/index.php/s/XXVKT5grAquXNqi/download?path=%2F&files=model_{}_ckpt.tar.gz"

STYLES = (
    "cezanne",
    "el-greco",
    "gauguin",
    "kandinsky",
    "kirchner",
    "monet",
    "morisot",
    "munch",
    "peploe",
    "picasso",
    "pollock",
    "roerich",
    "van-gogh",
)


def main(root="."):
    for style in STYLES:
        url = URL_FMTSTR.format(style)
        with get_tmp_dir() as tmp_dir:
            download_and_extract_archive(url, tmp_dir)
            file = locate_checkpoint(tmp_dir)
            tf_state_dict = load_tf_state_dict(file)
        torch_state_dict = convert_transformer(tf_state_dict)
        save_transformer(torch_state_dict, style, root)


def locate_checkpoint(root):
    for dirpath, _, _ in os.walk(root, topdown=False):
        ckpt = tf.train.get_checkpoint_state(dirpath)
        if ckpt is not None:
            return ckpt.model_checkpoint_path


def load_tf_state_dict(file):
    return OrderedDict(
        [
            (name, torch.from_numpy(tf.train.load_variable(file, name)).squeeze())
            for name, _ in tf.train.list_variables(file)
            if name.startswith(("encoder", "decoder")) and "Adam" not in name
        ]
    )


def convert_transformer(tf_state_dict):
    tf_state_dict = copy(tf_state_dict)
    torch_state_dict = OrderedDict()
    convert_encoder(tf_state_dict, "encoder", torch_state_dict, "encoder")
    convert_decoder(tf_state_dict, "decoder", torch_state_dict, "decoder")
    assert not tf_state_dict
    return torch_state_dict


def save_transformer(state_dict, style, root):
    name_parts = ["sanakoyeu_et_al_2018_transformer", style, "impl_params"]
    name = "__".join(name_parts)
    return save_state_dict(state_dict, name, root)


def convert_encoder(tf_state_dict, tf_key_prefix, torch_state_dict, torch_key_prefix):
    tf_key_fmtstr = make_tf_key(tf_key_prefix, "g_e{}")
    torch_key_fmtstr = make_torch_key(torch_key_prefix, "{}")

    convert_norm(
        tf_state_dict,
        tf_key_fmtstr.format(0),
        torch_state_dict,
        torch_key_fmtstr.format(0),
    )
    for idx in range(1, 6):
        convert_conv_block(
            tf_state_dict,
            tf_key_fmtstr.format(idx),
            torch_state_dict,
            torch_key_fmtstr.format(idx + 1),
        )
    return torch_state_dict


def convert_decoder(tf_state_dict, tf_key_prefix, torch_state_dict, torch_key_prefix):
    tf_key_fmtstr = make_tf_key(tf_key_prefix, "g_r{}")
    torch_key_fmtstr = make_torch_key(torch_key_prefix, "{}", "residual")
    num_residual_blocks = 9
    for idx in range(1, num_residual_blocks + 1):
        convert_residual_block(
            tf_state_dict,
            tf_key_fmtstr.format(idx),
            torch_state_dict,
            torch_key_fmtstr.format(idx - 1),
        )

    tf_key_fmtstr = make_tf_key(tf_key_prefix, "g_d{}")
    torch_key_fmtstr = make_torch_key(torch_key_prefix, "{}")
    for idx in range(1, 5):
        convert_upsample_conv_block(
            tf_state_dict,
            tf_key_fmtstr.format(idx),
            torch_state_dict,
            torch_key_fmtstr.format(num_residual_blocks + idx - 1),
        )

    convert_conv(
        tf_state_dict,
        make_tf_key(tf_key_prefix, "g_pred"),
        torch_state_dict,
        make_torch_key(torch_key_prefix, num_residual_blocks + 4),
    )
    return torch_state_dict


def _make_key(sep, *parts):
    return sep.join([str(part) for part in parts if part != ""])


def make_torch_key(*parts):
    return _make_key(".", *parts)


def make_tf_key(*parts):
    return _make_key("/", *parts)


def _convert(
    tf_state_dict,
    tf_key,
    torch_state_dict,
    torch_key,
    delete=True,
    converter=lambda x: x,
):
    tf_param = tf_state_dict[tf_key]
    torch_param = converter(tf_param)
    torch_state_dict[torch_key] = torch_param
    if delete:
        del tf_state_dict[tf_key]


def conv_weights_converter(tf_conv_weights):
    ndim = tf_conv_weights.dim()
    dims = (ndim - 1, ndim - 2, *range(ndim - 2))
    return tf_conv_weights.permute(*dims).contiguous()


def convert_conv(
    tf_state_dict, tf_key_prefix, torch_state_dict, torch_key_prefix, tf_key_postfix="c"
):
    _convert(
        tf_state_dict,
        make_tf_key(f"{tf_key_prefix}_{tf_key_postfix}", "Conv", "weights"),
        torch_state_dict,
        make_torch_key(torch_key_prefix, "weight"),
        converter=conv_weights_converter,
    )
    return torch_state_dict


def convert_deconv(
    tf_state_dict,
    tf_key_prefix,
    torch_state_dict,
    torch_key_prefix,
    tf_key_postfix="dc",
):
    _convert(
        tf_state_dict,
        make_tf_key(f"{tf_key_prefix}_{tf_key_postfix}", "conv2d", "Conv", "weights"),
        torch_state_dict,
        make_torch_key(torch_key_prefix, "weight"),
        converter=conv_weights_converter,
    )
    return torch_state_dict


def convert_norm(
    tf_state_dict,
    tf_key_prefix,
    torch_state_dict,
    torch_key_prefix,
    tf_key_postfix="bn",
):
    tf_key_prefix = f"{tf_key_prefix}_{tf_key_postfix}"
    _convert(
        tf_state_dict,
        make_tf_key(tf_key_prefix, "scale"),
        torch_state_dict,
        make_torch_key(torch_key_prefix, "weight"),
    )
    _convert(
        tf_state_dict,
        make_tf_key(tf_key_prefix, "offset"),
        torch_state_dict,
        make_torch_key(torch_key_prefix, "bias"),
    )
    return torch_state_dict


def convert_conv_block(
    tf_state_dict,
    tf_key_prefix,
    torch_state_dict,
    torch_key_prefix,
    tf_conv_key_postfix="c",
    tf_norm_key_postfix="bn",
):
    convert_conv(
        tf_state_dict,
        tf_key_prefix,
        torch_state_dict,
        make_torch_key(torch_key_prefix, 0),
        tf_key_postfix=tf_conv_key_postfix,
    )
    convert_norm(
        tf_state_dict,
        tf_key_prefix,
        torch_state_dict,
        make_torch_key(torch_key_prefix, 1),
        tf_key_postfix=tf_norm_key_postfix,
    )
    return torch_state_dict


def convert_residual_block(
    tf_state_dict, tf_key_prefix, torch_state_dict, torch_key_prefix
):
    torch_key_fmtstr = make_torch_key(torch_key_prefix, "{}")
    for idx in range(1, 3):
        convert_conv_block(
            tf_state_dict,
            tf_key_prefix,
            torch_state_dict,
            torch_key_fmtstr.format(idx - 1),
            tf_conv_key_postfix=f"c{idx}",
            tf_norm_key_postfix=f"bn{idx}",
        )


def convert_upsample_conv_block(
    tf_state_dict, tf_key_prefix, torch_state_dict, torch_key_prefix
):
    convert_deconv(
        tf_state_dict,
        tf_key_prefix,
        torch_state_dict,
        make_torch_key(torch_key_prefix, 0),
    )
    convert_norm(
        tf_state_dict,
        tf_key_prefix,
        torch_state_dict,
        make_torch_key(torch_key_prefix, 1),
    )
    return torch_state_dict


if __name__ == "__main__":
    root = path.join(path.dirname(__file__), "data", "models")
    os.makedirs(root, exist_ok=True)
    main(root)
