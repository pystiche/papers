import json
import os

import pytest

from torch import hub


@pytest.fixture(scope="package")
def github():
    def format(owner, repository, branch):
        return f"{owner}/{repository}:{branch}"

    default = format("pystiche", "papers", "main")

    if os.getenv("GITHUB_ACTIONS", False):
        context = json.loads(os.getenv("GITHUB_CONTEXT"))

        repository = context["repository"].split("/")[-1]
        event_name = context["event_name"]

        if event_name == "push":
            owner = context["repository_owner"]
            branch = context["event"]["ref"].rsplit("/", 1)[-1]
        elif event_name == "pull_request":
            label = context["event"]["pull_request"]["head"]["label"]
            owner, branch = label.split(":")
        elif event_name == "schedule":
            owner = context["repository_owner"]
            branch = context["ref"].rsplit("/", 1)[-1]
        else:
            return default

        return format(owner, repository, branch)

    return default


@pytest.fixture(scope="package", autouse=True)
def reload_github(github):
    hub._get_cache_or_reload(github, force_reload=True, verbose=False)
