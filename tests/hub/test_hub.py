import pytest

from torch import hub, nn


@pytest.fixture(scope="module")
def entry_points():
    return ("johnson_alahi_li_2016_transformer",)


def test_hub_entrypoints(github, entry_points):
    print("=" * 80)
    print("#" * 80)
    print(github)
    print("#" * 80)
    print("=" * 80)
    models = hub.list(github)
    assert set(models) == set(entry_points)


def test_hub_help_smoke(subtests, github, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.help(github, model), str)


def test_hub_load_smoke(subtests, github, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.load(github, model), nn.Module)
