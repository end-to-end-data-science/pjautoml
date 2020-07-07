from tests.utiltest import SimpElem
from paje.base.data import Data
import numpy as np
import pytest

@pytest.fixture
def simple_data():
    return Data('simple', X=np.array([[1, 2], [3, 4]]))


@pytest.fixture
def get_elements(simple_data):
    elem_a, elem_b, elem_d = (SimpElem(), SimpElem(), SimpElem())

    aaa = elem_a.build(oper='+')
    bbb = elem_b.build(oper='.')
    ccc = elem_a.build(oper='+')
    ddd = elem_d.build(oper='*')

    return (aaa, bbb, ccc, ddd)

