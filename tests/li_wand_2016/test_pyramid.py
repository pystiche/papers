import numpy as np

from pystiche.pyramid import OctaveImagePyramid
from pystiche_papers.li_wand_2016 import pyramid


def test_li_wand_2016_image_pyramid(subtests):

    configs = ((True, 100, 3), (False, 200, None))
    for (impl_params, num_steps, num_levels,) in configs:
        with subtests.test(impl_params=impl_params):
            if num_levels is None:
                num_levels = int(np.floor(np.log2(6))) + 1
            li_wand_pyramid = pyramid.li_wand_2016_image_pyramid(
                impl_params=impl_params, num_steps=num_steps, num_levels=num_levels,
            )
            assert isinstance(li_wand_pyramid, OctaveImagePyramid)

            with subtests.test("num_steps"):
                pyramid_num_steps = tuple(
                    level.num_steps for level in li_wand_pyramid._levels
                )
                assert pyramid_num_steps == (num_steps,) * len(li_wand_pyramid._levels)

            with subtests.test("num_levels"):
                assert len(li_wand_pyramid._levels) == num_levels
