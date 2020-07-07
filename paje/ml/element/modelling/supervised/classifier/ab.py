from math import *

from sklearn.ensemble import AdaBoostClassifier

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class AB(Classifier):
    def build_impl(self):
        self.model = AdaBoostClassifier(**self.config)
        raise Exception(
            'Working in progress... AB still with RF hyperparameters.')

    @classmethod
    def hps_impl(cls, data):
        cls.check_data(data)
        n_estimators = min([500, floor(sqrt(
            data.n_instances() * data.n_attributes()))])
        raise Exception(
            'Working in progress... AB still with RF hyperparameters.')
        data_for_speed = {
            'n_estimators': ['z', [2, 1000]],
            'max_depth': ['z', [2, data.n_instances()]]
        }  # Entre outros
        node = {'bootstrap': ['c', [True, False]],
               'min_impurity_decrease': ['r', [0, 1]],
               'max_leaf_nodes': ['o',
                                  [2, 3, 5, 8, 12, 17, 23,
                                   30, 38, 47, 57, 999999]],  # 999999 ~ None

               # For some reason, the interval [1, n_attributes] didn't work
               # for RF (maybe it is relative to a subset).
               'max_features': ['r', [0.001, 1]],

               # According to ValueError exception.
               'min_weight_fraction_leaf': ['r', [0, 0.5]],

               # Int (# of instances) is better than float
               # (proportion of instances) because different floats can collide
               # with a same int, making intervals of useless real values.
               'min_samples_leaf': ['z', [1, floor(data.n_instances() / 2)]],

               # Same reason as min_samples_leaf
               'min_samples_split': ['z', [2, floor(data.n_instances() / 2)]],

               'max_depth': ['z', [2, data.n_instances()]],

               # Docs say that this parameter is tree-specific,
               # but we cannot choose the tree.
               'criterion': ['c', ['gini', 'entropy']],

               # Only to set the default, not for search.
               'n_estimators': ['c', [n_estimators]]

               # See DT.py for more details about other settings.
               }
        return HPTree(node, children=[])
