# Contributing guide lines

We appreciate all contributions! If you are planning to contribute bug-fixes or
documentation improvements, please go ahead and open a
[pull request (PR)](https://github.com/pystiche/papers/pulls). If you are planning to
contribute new features, please open an
[issue](https://github.com/pystiche/papers/issues) and discuss the feature with us
first.

To start working on `pystiche-papers` clone the repository from GitHub and set up the
development environment

```shell
git clone https://github.com/pystiche/papers
cd papers
virtualenv .venv --prompt='(pystiche-papers-dev) '
source .venv/bin/activate
pip install doit
doit install
```

Every PR is subjected to multiple checks that it has to pass before it can be merged.
The checks are performed through [doit](https://pydoit.org/). Below you can find details
and instructions how to run the checks locally.

## Code format and linting

`pystiche-papers` uses [ufmt](https://ufmt.omnilib.dev/en/stable/) to format Python
code, and [flake8](https://flake8.pycqa.org/en/stable/) to enforce
[PEP8](https://www.python.org/dev/peps/pep-0008/) compliance.

Furthermore, `pystiche-papers` is [PEP561](https://www.python.org/dev/peps/pep-0561/)
compliant and checks the type annotations with [mypy](http://mypy-lang.org/) .

To automatically format the code, run

```shell
doit format
```

Instead of running the formatting manually, you can also add
[pre-commit](https://pre-commit.com/) hooks. By running

```shell
pre-commit install
```

once, an equivalent of `doit format` is run everytime you `git commit` something.

Everything that cannot be fixed automatically, can be checked with

```shell
doit lint
```

## Tests

`pystiche-papers` uses [pytest](https://docs.pytest.org/en/stable/) to run the test
suite. You can run it locally with

```sh
doit test
```

## Documentation

To build the documentation locally, run

```sh
doit doc
```
