import csv
from os import path

import pytest

from torch import hub

import pystiche_papers
from pystiche_papers.utils import str_to_bool


@pytest.mark.large_download
@pytest.mark.slow
def test_hub_sanakoyeu_et_al_2018_transformer(subtests, github):
    def configs():
        file = path.join(
            pystiche_papers.__path__[0], "sanakoyeu_et_al_2018", "model_urls.csv"
        )
        with open(file, "r",) as fh:
            for row in csv.DictReader(fh):
                row["impl_params"] = str_to_bool(row["impl_params"])
                del row["url"]
                yield row

    for config in configs():
        with subtests.test(**config):
            assert hub.load(github, "sanakoyeu_et_al_2018_transformer", **config)
