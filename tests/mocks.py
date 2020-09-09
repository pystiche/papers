import functools
import unittest.mock

import torch

__all__ = [
    "make_mock_target",
    "attach_method_mock",
    "patch_models_load_state_dict_from_url",
    "patch_multi_layer_encoder_loader",
    "mock_images",
]

DEFAULT_MOCKER = unittest.mock


def make_mock_target(*args, pkg="pystiche_papers"):
    return ".".join((pkg, *args))


def attach_method_mock(mock, method, mocker=DEFAULT_MOCKER, **attrs):
    if "name" not in attrs:
        attrs["name"] = f"{mock.name}.{method}()"

    method_mock = mocker.Mock(**attrs)
    mock.attach_mock(method_mock, method)


def patch_models_load_state_dict_from_url(mocker=DEFAULT_MOCKER):
    return mocker.patch(
        make_mock_target(
            "enc",
            "models",
            "utils",
            "ModelMultiLayerEncoder",
            "load_state_dict_from_url",
            pkg="pystiche",
        )
    )


@functools.lru_cache(maxsize=4)
def _load_multi_layer_encoder(loader, *args, **kwargs):
    multi_layer_encoder = loader(*args, **kwargs)
    multi_layer_encoder.trim = unittest.mock.Mock()
    return multi_layer_encoder


def patch_multi_layer_encoder_loader(
    targets,
    loader,
    patch_load_weights=True,
    clear_cache=False,
    setups=None,
    mocker=DEFAULT_MOCKER,
):
    if patch_load_weights:
        patch_models_load_state_dict_from_url(mocker=mocker)

    if clear_cache:
        _load_multi_layer_encoder.cache_clear()

    def new(*args, **kwargs):
        multi_layer_encoder = _load_multi_layer_encoder(loader, *args, **kwargs)
        multi_layer_encoder.empty_storage()
        return multi_layer_encoder

    if not isinstance(targets, list):
        targets = [targets]

    patches = tuple(mocker.patch(target, new) for target in targets)

    if setups is not None:
        if not isinstance(setups, list):
            setups = [setups]
        for setup in setups:
            args, kwargs = setup
            new(*args, **kwargs)

    return patches


def _make_image_mock(image=None, mocker=DEFAULT_MOCKER):
    if image is None:
        image = torch.rand(1, 3, 32, 32)
    mock = mocker.Mock()
    attach_method_mock(mock, "read", return_value=image)
    attach_method_mock(mock, "download")
    return mock


def mock_images(mocker, *args, **kwargs):
    image_mocks = {name: _make_image_mock(mocker=mocker) for name in args}
    image_mocks.update(
        {name: _make_image_mock(image, mocker=mocker) for name, image in kwargs.items()}
    )

    def getitem_side_effect(name):
        try:
            return image_mocks[name]
        except KeyError:
            return unittest.mock.DEFAULT

    images_mock = mocker.Mock()
    attach_method_mock(images_mock, "download", mocker=mocker)
    attach_method_mock(
        images_mock, "__getitem__", side_effect=getitem_side_effect, mocker=mocker
    )
    attach_method_mock(
        images_mock,
        "read",
        return_value={name: image.read() for name, image in image_mocks.items()},
        mocker=mocker,
    )

    return images_mock
