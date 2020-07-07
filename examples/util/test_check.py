import pytest
import numpy as np
from paje.util.check import check_is, check_min_max, check_int, check_float, \
    check_X_y

def test_checks(name="Att_name"):
    value = False
    value = check_is(name, float, 0.22)
    assert value == True

    value = False
    value = check_min_max(name, 10, 0, 10)
    assert value == True

    value = False
    value = check_int(name, 0, 0, 10)
    assert value == True

    value = False
    value = check_float(name, 1.0, 1.0, 1.0)
    assert value == True

    value = False
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([1,2])
    value = check_X_y(X, y)
    assert value == True


def test_check_raise(name="Att_name"):
    with pytest.raises(ValueError):
        check_is(name, float, 1)

    with pytest.raises(ValueError):
        check_min_max(name, 11, 0, 10)

    with pytest.raises(ValueError):
        check_int(name, -1, 0, 10)

    with pytest.raises(ValueError):
        check_float(name, 1.0, 0, 0.9)

    with pytest.raises(ValueError):
        X = [[1,2,3], [4,5,6]]
        y = [1,2]
        check_X_y(X, y)

    with pytest.raises(ValueError):
        X = np.array([[1,2,3], [4,5,6]])
        y = np.array([1,2,3])
        check_X_y(X, y)


