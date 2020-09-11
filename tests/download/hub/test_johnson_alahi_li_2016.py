import csv
from os import path

import pytest

from torch import hub

import pystiche_papers


@pytest.mark.large_download
@pytest.mark.slow
def test_hub_johnson_alahi_li_2016_transformer(subtests, github):
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
            assert hub.load(github, "johnson_alahi_li_2016_transformer", **config)
