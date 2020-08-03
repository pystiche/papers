import os

import pytest

from torch import hub


@pytest.fixture(scope="package")
def github():
    return os.getenv("PYSTICHE_HUB_GITHUB", default="pmeier/pystiche_papers:master")


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
