import os

import pytest

from torch import hub


def trace(arg):
    print("=" * 80)
    print("#" * 80)
    print(arg)
    print("#" * 80)
    print("=" * 80)


@pytest.fixture(scope="package")
def github():
    trace(f"GITHUB_ACTIONS: {os.getenv('GITHUB_ACTIONS')}")
    if os.getenv("GITHUB_ACTIONS"):
        owner_and_repo = os.getenv("GITHUB_REPOSITORY")

        branch_or_tag = os.getenv("GITHUB_HEAD_REF")
        trace(f"GITHUB_HEAD_REF: {os.getenv('GITHUB_HEAD_REF')}")
        is_pr = branch_or_tag is not None
        if not is_pr:
            branch_or_tag = os.getenv("GITHUB_REF").rsplit("/", 1)[1]

        trace(f"branch_or_tag: {branch_or_tag}")
        return f"{owner_and_repo}:{branch_or_tag}"
    else:
        return os.getenv("PYSTICHE_HUB_GITHUB", default="pmeier/pystiche_papers:master")


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
