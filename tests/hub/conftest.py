import pytest

from torch import hub


@pytest.fixture(scope="package")
def github():
    return "pmeier/pystiche_papers"


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
