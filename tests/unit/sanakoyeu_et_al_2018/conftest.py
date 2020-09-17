import pytest


@pytest.fixture(scope="package")
def styles():
    return (
        "cezanne",
        "el-greco",
        "gauguin",
        "kandinsky",
        "kirchner",
        "monet",
        "morisot",
        "munch",
        "peploe",
        "picasso",
        "pollock",
        "roerich",
        "van-gogh",
    )
