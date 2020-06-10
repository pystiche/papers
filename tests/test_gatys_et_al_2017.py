import unittest
from os import path

import pytest

from torchvision.datasets.utils import calculate_md5

from pystiche_papers import gatys_et_al_2017 as paper

from .utils import get_tmp_dir


class TestData(unittest.TestCase):
    @pytest.mark.large_download
    @pytest.mark.slow
    def test_gatys_et_al_2017_images(self):
        with get_tmp_dir() as root:
            images = paper.gatys_et_al_2017_images()
            images.download(root=root)

            for _, image in images:
                with self.subTest(image=image):
                    file = path.join(root, image.file)

                    self.assertTrue(
                        path.exists(file), msg=f"File {file} does not exist."
                    )

                    actual = calculate_md5(file)
                    desired = image.md5
                    self.assertEqual(
                        actual,
                        desired,
                        msg=(
                            f"The actual and desired MD5 hash of the image mismatch: "
                            f"{actual} != {desired}"
                        ),
                    )
