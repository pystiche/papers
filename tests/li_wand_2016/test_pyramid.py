import pystiche_papers.li_wand_2016 as paper


def test_image_pyramid(subtests, mocker):
    mock = mocker.patch("pystiche_papers.li_wand_2016._pyramid.pyramid")

    for impl_params, num_steps, num_levels in ((True, 100, 3), (False, 200, None)):
        with subtests.test(impl_params=impl_params):
            mock.reset()

            paper.image_pyramid(impl_params=impl_params)

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
                    if isinstance(num_levels, int):
                        assert pyramid_num_levels == num_levels
                    else:
                        assert isinstance(pyramid_num_levels, int)

                with subtests.test("min_edge_size"):
                    assert pyramid_min_edge_size == 64

                with subtests.test("edge"):
                    assert pyramid_edge == "long"
