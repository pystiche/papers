import contextlib
import hashlib
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from os import path
from urllib.parse import urljoin
from urllib.request import urlopen

import torch
from torch.legacy.nn import (
    MulConstant,
    ReLU,
    Sequential,
    SpatialBatchNormalization,
    SpatialConvolution,
    SpatialFullConvolution,
    SpatialReflectionPadding,
    Tanh,
)
from torch.utils.serialization import load_lua
from torch.utils.serialization.read_lua_file import TorchObject


def main(root="."):
    if torch.__version__ >= "1":
        raise RuntimeError

    with make_tempdir() as tempdir:
        for url in get_luatorch_urls():
            print(f"Processing {url}")
            input_file = download_file(url, tempdir)
            content = load_lua(input_file, unknown_classes=True)

            config = extract_configuration(content["opt"])
            for name, val in config.items():
                print(f"{name}: {val}")

            state_dict = extract_transformer_state_dict(content["model"])
            output_file = save_transformer(
                state_dict, config["style"], config["instance_norm"], root
            )
            print(f"Saved to {output_file}")
            print("-" * 80)


def get_luatorch_urls():
    def parse(content):
        lines = iter(content.splitlines())

        root_pattern = re.compile(r'^BASE_URL="(?P<root>[^"]+)"\d*$')
        for line in lines:
            match = root_pattern.match(line)
            if match is not None:
                root = match.group("root")
                break
        else:
            raise RuntimeError

        file_url_pattern = re.compile(r'^curl -O "[$]BASE_URL/(?P<url>[\w./]+)"$')
        for line in lines:
            match = file_url_pattern.match(line)
            if match is not None:
                yield urljoin(root, match.group("url"))

    url = "https://raw.githubusercontent.com/pmeier/fast-neural-style/master/models/download_style_transfer_models.sh"
    with urlopen(url) as response:
        content = response.read().decode("utf-8").strip()

    yield from parse(content)


def extract_configuration(param_dict):
    return OrderedDict(
        (
            ("style", extract_style(param_dict)),
            ("instance_norm", extract_instance_norm(param_dict)),
            ("style_image_size", param_dict["style_image_size"]),
            ("content_weight", extract_content_weight(param_dict)),
            ("style_weight", extract_style_weight(param_dict)),
            ("regularization_weight", extract_regularization_weight(param_dict)),
            ("num_iterations", param_dict["num_iterations"]),
        )
    )


def extract_style(params):
    return path.splitext(path.basename(params["style_image"]))[0]


def extract_instance_norm(params):
    try:
        return bool(params["use_instance_norm"])
    except KeyError:
        return False


def _extract_weight(params, type):
    key = f"{type}_weight"
    try:
        return params[key]
    except KeyError:
        weights = params[f"{key}s"]
        weight = weights[0]
        assert all(other == weight for other in weights[1:])
        return weight


def extract_style_weight(params):
    return _extract_weight(params, "style")


def extract_content_weight(params):
    return _extract_weight(params, "content")


def extract_regularization_weight(params):
    return params["tv_strength"]


def extract_transformer_state_dict(model):
    modules = iter(model.modules)
    state_dict = OrderedDict()

    next_module = extract_encoder_params(modules, state_dict)
    extract_decoder_params(next_module, modules, state_dict)

    return state_dict


def extract_encoder_params(modules, state_dict):
    prefix = "encoder"

    pad = next(modules)
    if not is_reflection_pad2d(pad):
        raise RuntimeError

    next_module = next(modules)
    encoder_idx = 1
    while not is_conv_transpose2d(next_module):
        if not is_residual_block(next_module):
            next_module = extract_conv_block_params(
                next_module, modules, state_dict, (prefix, encoder_idx)
            )
        else:
            next_module = extract_residual_block_params(
                next_module, modules, state_dict, (prefix, encoder_idx)
            )

        encoder_idx += 1

    return next_module


def extract_decoder_params(next_module, modules, state_dict):
    prefix = "decoder"

    decoder_idx = 0
    while is_conv_transpose2d(next_module):
        next_module = extract_conv_block_params(
            next_module, modules, state_dict, (prefix, decoder_idx)
        )
        decoder_idx += 1

    if not is_conv2d(next_module):
        raise RuntimeError
    next_module = extract_conv_params(
        next_module, modules, state_dict, (prefix, decoder_idx)
    )

    if not is_tanh(next_module):
        raise RuntimeError
    next_module = next(modules)

    if not is_multiplication(next_module):
        raise RuntimeError
    next_module = next(modules)

    if not is_total_variation_loss(next_module):
        raise RuntimeError
    try:
        next(modules)
    except StopIteration:
        pass
    else:
        raise RuntimeError


def extract_conv_params(next_module, modules, state_dict, prefixes):
    if not is_conv(next_module):
        raise RuntimeError

    key_fmt = state_dict_key(*prefixes, "{}")
    state_dict[key_fmt.format("weight")] = next_module.weight
    state_dict[key_fmt.format("bias")] = next_module.bias

    return next(modules)


def extract_norm_params(next_module, modules, state_dict, prefixes):
    if not is_norm(next_module):
        raise RuntimeError

    key_fmt = state_dict_key(*prefixes, "{}")
    state_dict[key_fmt.format("weight")] = next_module.weight
    state_dict[key_fmt.format("bias")] = next_module.bias

    running_mean = next_module.running_mean
    if running_mean is None:
        running_mean = torch.zeros_like(next_module.weight)
    state_dict[key_fmt.format("running_mean")] = running_mean

    running_var = next_module.running_var
    if running_var is None:
        running_var = torch.ones_like(next_module.weight)
    state_dict[key_fmt.format("running_var")] = running_var

    return next(modules)


def extract_conv_block_params(next_module, modules, state_dict, prefixes):
    next_module = extract_conv_params(next_module, modules, state_dict, (*prefixes, 0))
    next_module = extract_norm_params(next_module, modules, state_dict, (*prefixes, 1))

    if is_relu(next_module):
        next_module = next(modules)

    return next_module


def extract_residual_block_params(residual_block, modules, state_dict, prefixes):
    residual_modules = iter(residual_block.modules[0].modules[0].modules)
    next_residual_module = next(residual_modules)
    try:
        for residual_block_idx in range(2):
            next_residual_module = extract_conv_block_params(
                next_residual_module,
                residual_modules,
                state_dict,
                (*prefixes, "residual", residual_block_idx),
            )
    except StopIteration:
        pass

    return next(modules)


def state_dict_key(*parts):
    return ".".join([str(part) for part in parts])


def is_reflection_pad2d(module):
    return isinstance(module, SpatialReflectionPadding)


def is_conv2d(module):
    return isinstance(module, SpatialConvolution)


def is_conv_transpose2d(module):
    return isinstance(module, SpatialFullConvolution)


def is_conv(module):
    return is_conv2d(module) or is_conv_transpose2d(module)


def is_batch_norm(module):
    return isinstance(module, SpatialBatchNormalization)


def is_torch_object(module, type_name=None):
    torch_object = isinstance(module, TorchObject)
    if type_name is not None:
        torch_object = torch_object and module.torch_typename() == type_name
    return torch_object


def is_instance_norm(module):
    return is_torch_object(module, "nn.InstanceNormalization")


def is_norm(module):
    return is_batch_norm(module) or is_instance_norm(module)


def is_relu(module):
    return isinstance(module, ReLU)


def is_residual_block(module):
    return isinstance(module, Sequential)


def is_tanh(module):
    return isinstance(module, Tanh)


def is_multiplication(module):
    return isinstance(module, MulConstant)


def is_total_variation_loss(module):
    return is_torch_object(module, "nn.TotalVariation")


@contextlib.contextmanager
def make_tempdir(cleanup=True, **mkdtemp_kwargs):
    tempdir = tempfile.mkdtemp(**mkdtemp_kwargs)
    try:
        yield tempdir
    finally:
        if cleanup:
            shutil.rmtree(tempdir)


def download_file(url, root, name=None):
    if name is None:
        name = path.basename(url)
    file = path.join(root, name)
    with open(file, "wb") as fh:
        fh.write(urlopen(url).read())
    return file


def save_transformer(state_dict, style, instance_norm, root):
    name_parts = ["johnson_alahi_li_2016_transformer", style, "impl_params"]
    if instance_norm:
        name_parts.append("instance_norm")
    name = "__".join(name_parts)
    return save_state_dict(state_dict, name, root)


def save_state_dict(
    state_dict,
    name,
    root,
    to_cpu: bool = True,
    hash_len: int = 8,
    ext: str = ".pth",
) -> str:
    if to_cpu:
        state_dict = OrderedDict(
            [(key, tensor.cpu()) for key, tensor in state_dict.items()]
        )

    with make_tempdir() as tempdir:
        tmp_file = path.join(tempdir, "tmp")
        torch.save(state_dict, tmp_file)
        sha256 = get_sha256_hash(tmp_file)

        file = path.join(root, f"{name}-{sha256[:hash_len]}{ext}")
        shutil.move(tmp_file, file)

    return file


def get_sha256_hash(file: str, chunk_size: int = 4096) -> str:
    hasher = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    root = path.join(path.dirname(__file__), "data", "models")
    os.makedirs(root, exist_ok=True)
    main(root)
