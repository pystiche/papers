import configparser
from os import path

try:
    import yaml
    from pytorch_wheel_installer.core import find_links
except ImportError:
    msg = "Please install pyyaml and pytorch_wheel_selector prior to running this."
    raise RuntimeError(msg)


def main(
    project_root=None, language=None, file="requirements-rtd.txt",
):
    if project_root is None:
        project_root = path.abspath(path.join(path.dirname(__file__), ".."))

    deps = extract_docs_deps(project_root)

    if language is None:
        language = extract_language_from_rtd_config(project_root)

    deps.extend(find_pytorch_wheel_links(language))

    with open(file, "w") as fh:
        fh.write("\n".join(deps) + "\n")


def extract_docs_deps(root, file="tox.ini", testenv="testenv:docs"):
    config = configparser.ConfigParser()
    config.read(path.join(root, file))
    deps = config[testenv]["deps"].strip().split("\n")
    # TODO: remove this when pystiche_papers has pystiche as a requirement
    deps.remove("git+https://github.com/pmeier/pystiche")
    return deps


def extract_language_from_rtd_config(root, file=".readthedocs.yml"):
    with open(path.join(root, file)) as fh:
        data = yaml.load(fh, Loader=yaml.FullLoader)

    python_version = str(data["python"]["version"])
    return f"py{python_version.replace('.', '')}"


def find_pytorch_wheel_links(
    language, distributions=("torch", "torchvision"), backend="cpu", platform="linux",
):
    return find_links(distributions, backend, language, platform)


if __name__ == "__main__":
    main()
