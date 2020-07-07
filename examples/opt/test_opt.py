"""
File: test_opt.py
Author: E. Alcobaca
Email: e.alcobaca@gmail.com
Github: https://github.com/ealcobaca
Description: Optimization test module
"""

import numpy as np
from paje.opt.random_search import RandomSearch
from paje.opt.hp_space import HPSpace


class TesteRandomSearch:
    """Test to the random search method

    """

    @staticmethod
    def my_func():
        """ function that return a sample of an attribute
        """
        return np.random.randint(1, 10)

    @staticmethod
    def making_space():
        """ Method that builds
        """
        hps = HPSpace(name="root")
        hps.add_axis(hps, "x1", 'c', 0, 5, ['0', '10', '15', '20', '25'])

        hpb1 = hps.new_branch(hps, "b1")
        hps.add_axis(hpb1, "x2", 'r', 0, 10, np.random.ranf)
        hps.add_axis(hpb1, "x3", 'z', -2, 10, np.random.ranf)

        hpb2 = hps.new_branch(hps, "b2")
        hps.add_axis(hpb2, "x4", 'f', None, None, TesteRandomSearch.my_func)

        hps.print(data=True)

        return hps

    @staticmethod
    def objective(*argv, **kwargs):
        """Test objective function
        """
        print("*argv --> {0} \n **kwargs --> {1}".format(argv, kwargs))

        aux = 10000
        x_3 = kwargs.get('x3')
        x_1 = kwargs.get('x1')
        x_4 = kwargs.get('x4')
        a = kwargs.get('a')
        # print(a)

        if x_4 is None:
            x_2 = kwargs.get('x2')
            aux = int(x_1) + x_2 + x_3 + 1
        elif x_3 is None:
            aux = int(x_1) * x_4
        else:
            print("Some error occur")

        return aux

    @staticmethod
    def test_randomsearch_seq():
        """ Method to random search test
        """
        np.random.seed(0)
        rans = RandomSearch(TesteRandomSearch.making_space(),
                            max_iter=10, n_jobs=1)

        result = rans.fmin(TesteRandomSearch.objective)
        result_test = (0, {'x1': '0', 'x4': 4})
        assert result == result_test

        np.random.seed(0)
        a = "teste"
        result = rans.fmin(TesteRandomSearch.objective, a=a)
        result_test = (0, {'x1': '0', 'x4': 4})
        assert result == result_test

    @staticmethod
    def test_randomsearch_par():
        """ Method to random search test
        """
        np.random.seed(0)
        rans = RandomSearch(TesteRandomSearch.making_space(),
                            max_iter=10, n_jobs=4)

        result = rans.fmin(TesteRandomSearch.objective)
        result_test = (0, {'x1': '0', 'x4': 4})
        assert result == result_test

        np.random.seed(0)
        a = "teste"
        result = rans.fmin(TesteRandomSearch.objective, a=a)
        result_test = (0, {'x1': '0', 'x4': 4})
        assert result == result_test
