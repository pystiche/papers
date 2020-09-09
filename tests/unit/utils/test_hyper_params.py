import pytest

from pystiche_papers.utils import HyperParameters


@pytest.fixture
def params():
    return {name: val for val, name in enumerate("abc")}


def test_HyperParameters_getattr(params):
    hyper_parameters = HyperParameters(**params)

    for name, val in params.items():
        assert getattr(hyper_parameters, name) == val


def test_HyperParameters_getattr_no_attribute():
    hyper_parameters = HyperParameters()

    with pytest.raises(AttributeError):
        hyper_parameters.unknown_attribute


def test_HyperParameters_contains(params):
    hyper_parameters = HyperParameters(**params)

    for name, val in params.items():
        assert name in hyper_parameters


def test_HyperParameters_properties(params):
    hyper_parameters = HyperParameters(**params)

    assert dict(hyper_parameters.properties()) == params


def test_HyperParameters_named_children(params):
    hyper_parameters = HyperParameters(**params)

    named_children = tuple(hyper_parameters.named_children())
    assert not named_children

    name = tuple(params.keys())[0]
    val = HyperParameters()
    setattr(hyper_parameters, name, val)

    named_children = tuple(hyper_parameters.named_children())
    assert len(named_children) == 1
    assert named_children[0][0] == name
    assert named_children[0][1] is val