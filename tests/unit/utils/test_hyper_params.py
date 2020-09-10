from copy import copy, deepcopy

import pytest

from pystiche_papers.utils import HyperParameters


@pytest.fixture
def params():
    return {name: val for val, name in enumerate("abc")}


def test_HyperParameters_getattr(params):
    hyper_parameters = HyperParameters(**params)

    for name, val in params.items():
        assert getattr(hyper_parameters, name) == val


def test_HyperParameters_getattr_unknown_attribute():
    hyper_parameters = HyperParameters()

    with pytest.raises(AttributeError):
        getattr(hyper_parameters, "unknown_attribute")


def test_HyperParameters_delattr(params):
    hyper_parameters = HyperParameters(**params)

    for name in params.keys():
        getattr(hyper_parameters, name)

        delattr(hyper_parameters, name)

        with pytest.raises(AttributeError):
            getattr(hyper_parameters, name)


def test_HyperParameters_delattr_unknown_attr(params):
    hyper_parameters = HyperParameters()

    with pytest.raises(AttributeError):
        delattr(hyper_parameters, "unknown_attribute")


def test_HyperParameters_contains(params):
    hyper_parameters = HyperParameters(**params)

    for name, val in params.items():
        assert name in hyper_parameters


def test_HyperParameters_copy():
    x = HyperParameters(param=object(), sub_param=HyperParameters(param=object()))
    y = copy(x)

    assert isinstance(y, HyperParameters)
    assert y is not x
    assert y.param is x.param
    assert y.sub_param is not x.sub_param
    assert y.sub_param.param is x.sub_param.param


def test_HyperParameters_deepcopy():
    x = HyperParameters(param=object(), sub_param=HyperParameters(param=object()))
    y = deepcopy(x)

    assert isinstance(y, HyperParameters)
    assert y is not x
    assert y.param is not x.param
    assert y.sub_param is not x.sub_param
    assert y.sub_param.param is not x.sub_param.param


def test_HyperParameters_new_similar(subtests, params):
    x = HyperParameters(**params)
    new = -1
    y = x.new_similar(a=new)

    assert isinstance(y, HyperParameters)

    with subtests.test("old"):
        del params["a"]
        for name, value in params.items():
            assert getattr(y, name) == value

    with subtests.test("new"):
        assert y.a == new


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
