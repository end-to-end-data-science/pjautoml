from paje.automl.composer.pipeline import Pipeline
import numpy as np
import pytest

from paje.searchspace.configspace import HPTree


@pytest.fixture
def get_pipeline_of_elem(get_elements, simple_data):
    aaa, bbb, ccc, ddd = get_elements
    data = simple_data

    compr = Pipeline(components=[aaa, bbb, ccc, ddd])
    configs = [
        {'name': aaa.name, 'oper': '+'},
        {'name': bbb.name, 'oper': '.'},
        {'name': ccc.name, 'oper': '+'},
        {'name': ddd.name, 'oper': '*'}
    ]

    mycompr = compr.build(configs=configs)
    data_apply = mycompr.apply(data)
    data_use = mycompr.use(data)

    return (mycompr, data_apply, data_use, data)


@pytest.fixture
def get_pipeline_of_pipeline(get_elements):
    aaa, bbb, ccc, ddd = get_elements

    compr1 = Pipeline(components=[aaa, bbb])
    compr2 = Pipeline(components=[ccc, ddd])
    compr3 = Pipeline(components=[compr1, compr2])

    config = {
        # 'name': compr3.name,
        'configs': [
            {
                # 'name': compr1.name,
                'configs': [
                    {'oper': '+'},
                    {'oper': '.'}
                ]
            },
            {
                # 'name': compr2.name,
                'configs': [
                    {'oper': '+'},
                    {'oper': '*'}
                ]
            }
        ]
    }

    return compr3.build(**config)


def test_apply_use(get_pipeline_of_elem):
    _, data_apply, data_use, data = get_pipeline_of_elem

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


def test_apply_use_pipeline(simple_data, get_pipeline_of_pipeline):
    mycompr = get_pipeline_of_pipeline
    data = simple_data

    data_apply = mycompr.apply(data)
    data_use = mycompr.use(data)

    test_apply_use((mycompr, data_apply, data_use, data))


def test_tree_pipeline(get_pipeline_of_pipeline):
    mycompr = get_pipeline_of_pipeline

    # teste pipeline of pipeline tree

    end = ('EndPipeline', {})
    pip = ('Pipeline', {})
    ele = ('SimpElem', {'oper': ['c', ['+', '-', '*', '.']]})

    def rec(nnd, child=None):
        l = [] if child is None else [child]
        name, node = nnd
        return HPTree(name=name, node=node, children=l)

    aux = rec(pip,
              rec(pip,
                  rec(ele,
                      rec(ele,
                          rec(end,
                              rec(pip,
                                  rec(ele,
                                      rec(ele,
                                          rec(end,
                                              rec(end))))))))))

    assert str(aux) == str(mycompr.cs())
