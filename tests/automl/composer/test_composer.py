from paje.automl.composer.composer import Composer
from paje.base.data import Data
import numpy as np
import pytest


class SimpCompr(Composer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_build = 0
        self.nothing = 0

    def build_impl(self):
        self.count_build += 1
        self.nothing = 1

    def cs_impl(self):
        pass


@pytest.fixture
def get_composer(get_elements, simple_data):
    aaa, bbb, ccc, ddd = get_elements
    data = simple_data

    compr = SimpCompr(components=[aaa, bbb, ccc, ddd])
    mycompr = compr.build()
    data_apply = mycompr.apply(data)
    data_use = mycompr.use(data)

    return (aaa, bbb, ccc, ddd, mycompr, data_apply, data_use, data)


def test_count_build_apply_use(get_composer):
    # test count
    aaa, bbb, ccc, ddd, mycompr, _, _, data = get_composer

    assert aaa.count_buid == aaa.count_apply == aaa.count_use == 1
    assert bbb.count_buid == bbb.count_apply == bbb.count_use == 1
    assert ccc.count_buid == ccc.count_apply == ccc.count_use == 1
    assert ddd.count_buid == ddd.count_apply == ddd.count_use == 1

    mycompr.use(data)
    mycompr.apply(data)

    # another apply and use
    assert aaa.count_buid == 1
    assert bbb.count_buid == 1
    assert ccc.count_buid == 1
    assert ddd.count_buid == 1

    assert aaa.count_apply == aaa.count_use == 2
    assert bbb.count_apply == bbb.count_use == 2
    assert ccc.count_apply == ccc.count_use == 2
    assert ddd.count_apply == ddd.count_use == 2


def test_apply_use(get_composer):
    aaa, _, _, _, _, data_apply, data_use, data = get_composer

    # sequence made in apply
    X = data.X
    X_aaa = X + X
    X_bbb = np.dot(X_aaa, X_aaa)
    X_ccc = X_bbb + X_bbb
    X_ddd = X_ccc * X_ccc
    assert np.allclose(X_ddd, data_apply.X)

    # sequence made in use
    X += X_aaa
    X += X_bbb
    X += X_ccc
    X += X_ddd
    assert np.allclose(X, data_use.X)
