from sklearn.svm import SVC

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class SVMC(Classifier): #, JSONEncoder):
    # def default(self, o):
    #     if isinstance(o, SVMC):
    #         print(o)
    #         return o.__name__
    #     else:
    #         return JSONEncoder.default(self, o)

    def build_impl(self):
        self.model = SVC(**self.config)

    @classmethod
    def cs_impl(self):
        # todo: set random seed; set 'cache_size'
        node = {
            'C': ['r', [0.0001, 100]],
            'shrinking': ['c', [True, False]],
            'probability': ['c', [False]],
            'tol': ['o',
                    [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                     1000, 10000]],
            'class_weight': ['c', [None, 'balanced']],
            # 'verbose': [False],
            'max_iter': ['c', [1000000]],
            'decision_function_shape': ['c', ['ovr', 'ovo']]
        }

        kernel_linear = HPTree({'kernel': ['c', ['linear']]}, children=[])

        kernel_poly = HPTree({
            'kernel': ['c', ['poly']],
            'degree': ['z', [0, 10]],
            'coef0': ['r', [0.0, 100]],
        }, children=[])

        kernel_rbf = HPTree({'kernel': ['c', ['rbf']]}, children=[])

        kernel_sigmoid = HPTree({
            'kernel': ['c', ['sigmoid']],
            'coef0': ['r', [0.0, 100]],
        }, children=[])

        kernel_nonlinear = HPTree({'gamma': ['r', [0.00001, 100]]},
                                  children=[kernel_poly, kernel_rbf,
                                            kernel_sigmoid])

        return HPTree(node, children=[kernel_linear, kernel_nonlinear])
