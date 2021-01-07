import itertools

import pytest

from tests import utils

impl_params_and_instance_norm = utils.parametrize_data(
    ("impl_params", "instance_norm"),
    *[
        pytest.param(impl_params, instance_norm)
        for impl_params, instance_norm in itertools.product(
            (True, False), (True, False)
        )
    ],
)
