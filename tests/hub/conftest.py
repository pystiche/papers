import json
import os

import pytest

from torch import hub


@pytest.fixture(scope="package")
def github():
    def format(owner, repository, branch):
        return f"{owner}/{repository}:{branch}"

    if os.getenv("GITHUB_ACTIONS", False):
        context = json.loads(os.getenv("GITHUB_CONTEXT"))

        is_pr = context["event_name"] == "pull_request"
        repository = context["repository"].split("/")[-1]

        if is_pr:
            label = context["event"]["pull_request"]["head"]["label"]
            owner, branch = label.split(":")
        else:
            owner = context["repository_owner"]
            branch = context["event"]["ref"].rsplit("/", 1)[-1]
        return format(owner, repository, branch)
    else:
        return format("pmeier", "pystiche_papers", "master")


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
