from catboost import CatBoostClassifier

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class CB(Classifier):
    def __init__(self, verbose=False, storage=None, show_warns=True):
        self.verbose = verbose
        super().__init__(storage=storage, show_warns=show_warns)

    def use_impl(self, data):
        # TODO: catboost seems to return a matrix instead of a vector; check
        #  if this solution applies; Saulo Guedes says it might be predicting
        #  for each tree.
        return data.updated(self, z=self.model.predict(data.X).flatten())

    def build_impl(self):
        self.model = CatBoostClassifier(**self.config,
                                        verbose=self.verbose)

    @classmethod
    def cs_impl(self):
        node = {
            'iterations': ['c', [100, 500, 1000, 3000, 5000]],
            'learning_rate': ['r', [0.000001, 1.0]],
            'depth': ['z', [1, 15]],
            'l2_leaf_reg': ['r', [0.01, 99999]],
            'loss_function': ['c', ['MultiClass']],
            'border_count': ['z', [1, 255]],
            'thread_count': ['c', [-1]]
        }

        return HPTree(node, children=[])
