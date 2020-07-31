import csv
from os import path

import pytest

from torch import hub, nn

import pystiche_papers

GITHUB = "pmeier/pystiche_papers:hub"


@pytest.fixture(scope="module", autouse=True)
def reload_github():
    hub._get_cache_or_reload(GITHUB, force_reload=True, verbose=False)


@pytest.fixture(scope="module")
def entry_points():
    return ("johnson_alahi_li_2016_transformer",)


def test_hub_entrypoints(entry_points):
    models = hub.list(GITHUB)
    assert set(models) == set(entry_points)


def test_hub_help_smoke(subtests, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.help(GITHUB, model), str)


def test_hub_load_smoke(subtests, entry_points):
    for model in entry_points:
        with subtests.test(model):
            assert isinstance(hub.load(GITHUB, model), nn.Module)


def test_hub_johnson_alahi_li_2016_transformer(subtests):
    def str_to_bool(string: str) -> bool:
        return string.lower() == "true"

    def configs():
        file = path.join(
            pystiche_papers.__path__[0], "johnson_alahi_li_2016", "model_urls.csv"
        )
        with open(file, "r",) as fh:
            for row in csv.DictReader(fh):
                row["impl_params"] = str_to_bool(row["impl_params"])
                row["instance_norm"] = str_to_bool(row["instance_norm"])
                del row["url"]
                yield row

    for config in configs():
        with subtests.test(**config):
            assert hub.load(GITHUB, "johnson_alahi_li_2016_transformer", **config)
