import numpy as np

from pystiche.pyramid import OctaveImagePyramid
from pystiche_papers.li_wand_2016 import pyramid


def test_li_wand_2016_image_pyramid(subtests, mocker):

    configs = ((True, 100, 3), (False, 200, None))
    for (impl_params, num_steps, num_levels,) in configs:
        with subtests.test(impl_params=impl_params):

            if num_levels is None:
                num_levels = int(np.floor(np.log2(6))) + 1
            mock = mocker.patch("pystiche.pyramid.pyramid.OctaveImagePyramid")
            pyramid.li_wand_2016_image_pyramid(impl_params=impl_params)

            for call in mock.call_args_list:
                args, kwargs = call
                pyramid_max_edge_size, pyramid_num_steps = args
                pyramid_num_levels = kwargs["num_levels"]
                pyramid_min_edge_size = kwargs["min_edge_size"]
                pyramid_edge = kwargs["edge"]

                with subtests.test("max_edge_size"):
                    assert pyramid_max_edge_size == 384

                with subtests.test("num_steps"):
                    assert pyramid_num_steps == num_steps

                with subtests.test("num_levels"):
                    assert pyramid_num_levels == num_levels

                with subtests.test("min_edge_size"):
                    assert pyramid_min_edge_size == 64

                with subtests.test("edge"):
                    assert pyramid_edge == "long"
