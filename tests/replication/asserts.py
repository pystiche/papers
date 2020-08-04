from os import path

__all__ = ["assert_dir_exists"]


def assert_dir_exists(dir):
    assert path.exists(dir), f"'{dir}' does not exist."
    assert path.isdir(dir), f"'{dir}' is not a directory."
