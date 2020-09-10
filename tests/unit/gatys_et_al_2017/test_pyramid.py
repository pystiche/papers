import pystiche_papers.gatys_et_al_2017 as paper
from pystiche import pyramid


def test_image_pyramid(subtests):
    image_pyramid = paper.image_pyramid()
    assert isinstance(image_pyramid, pyramid.ImagePyramid)

    levels = tuple(iter(image_pyramid))
    assert len(levels) == 2

    edge_sizes, num_steps = zip(
        *[(level.edge_size, level.num_steps) for level in levels]
    )
    hyper_parameters = paper.hyper_parameters().image_pyramid

    with subtests.test("edge_sizes"):
        assert edge_sizes == hyper_parameters.edge_sizes

    with subtests.test("num_steps"):
        assert num_steps == hyper_parameters.num_steps
