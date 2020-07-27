from typing import Dict, cast


from torch import nn

from pystiche_papers.ulyanov_et_al_2016 import modules


def test_SequentialWithOutChannels(subtests):
    sequential_modules = [nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1)]
    for out_channel_name in (
        None,
        1,
        "1",
    ):  # TODO: Rename out_channel_name is not implemented
        module = modules.SequentialWithOutChannels(
            *sequential_modules, out_channel_name=out_channel_name
        )

        with subtests.test("out_channel_name"):
            if out_channel_name is None:
                desired_out_channel_name = "1"
            elif isinstance(out_channel_name, int):
                desired_out_channel_name = str(out_channel_name)
            else:
                desired_out_channel_name = out_channel_name

            assert (
                tuple(cast(Dict[str, nn.Module], module._modules).keys())[-1]
                == desired_out_channel_name
            )

        with subtests.test("out_channels"):
            assert module.out_channels == sequential_modules[-1].out_channels