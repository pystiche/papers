import random
import sys
from os import path

import numpy as np

import torch

from pystiche.image import write_image as _write_image
from pystiche_papers.utils import make_reproducible


def write_image(root, name, image_size, channels=3, ext=".png"):
    _write_image(torch.rand(channels, *image_size), path.join(root, name + ext))


def generate_small_images(root, name="small", image_size=(32, 32)):
    for idx in range(3):
        write_image(root, f"{name}_{idx}", image_size)

    landscape_size = (image_size[0] // 2, image_size[1])
    write_image(root, f"{name}_landscape", landscape_size)

    portrait_size = (image_size[0], image_size[1] // 2)
    write_image(root, f"{name}_portrait", portrait_size)


def generate_medium_image(root, name="medium", image_size=(128, 128)):
    write_image(root, name, image_size)


def generate_large_image(root, name="large", image_size=(256, 256)):
    write_image(root, name, image_size)


def save_versions(root):
    versions = {
        "python": sys.version[:5],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    with open(path.join(root, "versions.txt"), "w") as fh:
        fh.write(
            "\n".join(
                [f"{package}=={version}" for package, version in versions.items()]
            )
            + "\n"
        )


def save_rng_states(root):
    rng_states = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    torch.save(rng_states, path.join(root, "rng_states.bin"))


def main(root):
    make_reproducible()
    save_versions(root)
    save_rng_states(root)

    generate_small_images(root)
    generate_medium_image(root)
    generate_large_image(root)


if __name__ == "__main__":
    root = path.abspath(path.dirname(__file__))
    main(root)
