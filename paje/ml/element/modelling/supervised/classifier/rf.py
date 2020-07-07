from math import *
from sklearn.ensemble import RandomForestClassifier

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class RF(Classifier):
    def build_impl(self):
        # TODO: set n_jobs in constructor
        self.model = RandomForestClassifier(**self.config)

    @classmethod
    def cs_impl(self):
        n_estimators = [100, 500, 1000, 3000, 5000]
        node = {'bootstrap': ['c', [True, False]],
               'criterion': ['c', ['gini', 'entropy']],
               'max_features': ['c', ['auto', 'sqrt', 'log2', None]],
               'min_impurity_decrease': ['r', [0, 0.2]],
               'min_samples_split': ['r', [1e-6, 0.3]],
               'min_samples_leaf': ['r', [1e-6, 0.3]],
               'min_weight_fraction_leaf': ['r', [0, 0.3]],
               'max_depth': ['z', [2, 1000]],
               'n_estimators': ['c', n_estimators],
               }
        return HPTree(node, children=[])

    @classmethod
    def tree_impl_back(cls, data):
        cls.check_data(data)
        n_estimators = min(
            [500, floor(sqrt(data.n_instances() * data.n_attributes()))])

        data_for_speed = {
            'n_estimators': ['z', [2, 1000]],
            'max_depth': ['z', [2, data.n_instances()]]}  # Entre outros
        node = {'bootstrap': ['c', [True, False]],
               'min_impurity_decrease': ['r', [0, 1]],
               'max_leaf_nodes': ['o', [2, 3, 5, 8, 12, 17, 23, 30, 38, 47, 57,
                                        999999]],  # 999999 ~ None
               'max_features': ['r', [0.001, 1]],
               # For some reason, the interval [1, n_attributes] didn't work for
               # RF (maybe it is relative to a subset).
               'min_weight_fraction_leaf': ['r', [0, 0.5]],
               # According to ValueError exception.
               'min_samples_leaf': ['z', [1, floor(data.n_instances() / 2)]],
               # Int (# of instances) is better than float
               # (proportion of instances) because different floats can collide
               # to a same int, making intervals of useless real values.
               'min_samples_split': ['z', [2, floor(data.n_instances() / 2)]],
               # Same reason as min_samples_leaf
               'max_depth': ['z', [2, data.n_instances()]],
               'criterion': ['c', ['gini', 'entropy']],
               # Docs say that this parameter is tree-specific,
               # but we cannot choose the tree.
               'n_estimators': ['c', [n_estimators]],
               # Only to set the default, not for search.
               # See DT.py for more details about other settings.
               }
        return HPTree(node, children=[])
