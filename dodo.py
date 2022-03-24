import itertools
import os
import pathlib
import shlex

from doit import task_params

from doit.action import CmdAction


HERE = pathlib.Path(__file__).parent
PACKAGE_NAME = "pystiche_papers"

CI = os.environ.get("CI") == "true"

DOIT_CONFIG = dict(
    verbosity=2,
    backend="json",
    default_tasks=[
        "lint",
        "test",
        "publishable",
    ],
)


def do(*cmd, cwd=HERE):
    if len(cmd) == 1 and callable(cmd[0]):
        cmd = cmd[0]
    else:
        cmd = list(itertools.chain.from_iterable(shlex.split(part) for part in cmd))
    return CmdAction(cmd, shell=False, cwd=cwd)


def task_install():
    """Installs all development requirements and pystiche-papers in development mode"""
    yield dict(
        name="dev",
        file_dep=[HERE / "requirements-dev.txt"],
        actions=[do("pip install -r requirements-dev.txt")],
    )
    yield dict(
        name="project",
        actions=[
            do("pip install --pre light-the-torch"),
            do("ltt install -e ."),
        ],
    )


def task_format():
    """Auto-formats all project files"""
    return dict(
        actions=[
            do("pre-commit run --all-files"),
        ]
    )


def task_lint():
    """Lints all project files"""
    yield dict(
        name="flake8",
        actions=[
            do("flake8 --config=.flake8"),
        ],
    )
    yield dict(
        name="mypy",
        actions=[
            do("mypy --config-file=mypy.ini"),
        ],
    )


@task_params(
    [
        dict(
            name="subset",
            long="subset",
            type=str,
            choices=[
                (subset, "")
                for subset in ["all", "unit", "replication", "download", "doc"]
            ],
            default="all",
        ),
        dict(
            name="large_download",
            long="large-download",
            type=bool,
            default=True,
        ),
    ]
)
def task_test(subset, large_download):
    """Runs the test suite"""
    pytest = dict(
        name="pytest",
        actions=[
            do(
                "pytest -c pytest.ini",
                f"--cov-report={'xml' if CI else 'term'}",
                *[f"tests/{subset}"] if subset != "all" else [],
                *["--skip-large-download"] if not large_download else [],
            )
        ],
    )
    sphinx = dict(
        name="sphinx",
        actions=[
            do(
                "sphinx-build -b doctest -W --keep-going source build",
                cwd=HERE / "docs",
            ),
        ],
    )

    if subset == "all":
        yield pytest
        yield sphinx
    elif subset == "doc":
        yield sphinx
    else:
        yield pytest


@task_params(
    [
        dict(
            name="builder",
            long="builder",
            short="b",
            type=str,
            choices=[("html", ""), ("latex", "")],
            default="html",
        ),
    ]
)
def task_doc(builder):
    """Builds the documentation"""
    return dict(
        actions=[
            do(f"sphinx-build -b {builder} source build/{builder}", cwd=HERE / "docs")
        ]
    )


def task_build():
    """Builds the source distribution and wheel"""
    return dict(
        actions=[
            do("python -m build ."),
        ],
        clean=[
            do(f"rm -rf build dist {PACKAGE_NAME}.egg-info"),
        ],
    )


def task_publishable():
    """Checks if metadata is correct"""
    yield dict(
        name="twine",
        actions=[
            # We need the lambda here to lazily glob the files in dist/*, since they
            # are only created by the build task rather than when this task is
            # created.
            do(lambda: ["twine", "check", *list((HERE / "dist").glob("*"))]),
        ],
        task_dep=["build"],
    )
    yield dict(
        name="check-wheel-contents",
        actions=[
            do("check-wheel-contents dist"),
        ],
        task_dep=["build"],
    )


def task_publish():
    """Publishes to PyPI"""
    # TODO: check if env vars are set
    return dict(
        # We need the lambda here to lazily glob the files in dist/*, since they are
        # only created by the build task rather than when this task is created.
        actions=[
            do(lambda: ["twine", "upload", *list((HERE / "dist").glob("*"))]),
        ],
        task_dep=[
            "lint",
            "test",
            "publishable",
        ],
    )
