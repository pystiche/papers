import os

import pytest

from torch import hub


@pytest.fixture(scope="package")
def github():
    if os.getenv("GITHUB_ACTIONS"):
        # FIXME: this environment variable does not contain the owner of the repository
        #  but rather the person who triggered the workflow.
        #  See https://github.com/pmeier/pystiche_papers/issues/116 for details
        owner = os.getenv("GITHUB_ACTOR")

        branch_or_tag = os.getenv("GITHUB_HEAD_REF")
        is_pr = bool(branch_or_tag)
        if not is_pr:
            branch_or_tag = os.getenv("GITHUB_REF").rsplit("/", 1)[1]

        return f"{owner}/pystiche_papers:{branch_or_tag}"
    else:
        return os.getenv("PYSTICHE_HUB_GITHUB", default="pmeier/pystiche_papers:master")


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
