from itertools import combinations

import numpy as np
import pandas as pd

from paje.ml.element.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import t_score
import math

class FilterTScore(Filter):
    """  """
    def apply_impl(self, data):
        X, y = data.Xy

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes

        self.apply_t_score(X, y)
        self._nro_features = math.ceil(self.ratio * X.shape[1])

        return self.use_impl(data)

    def comb_idx(self, n, k):
        return np.array(list(combinations(range(n), k)))

    def apply_t_score(self, X, y):
        cat = np.unique(y)
        cat_len = len(cat)
        idx_cat = [y == i for i in cat]
        aux = []
        # rank_point = np.zeros(X.shape[1])

        for a, b in self.comb_idx(cat_len, 2):
            idx = np.logical_or(idx_cat[a], idx_cat[b])
            score = t_score.t_score(X[idx], y[idx])
            aux.append(score)

        # If there is more one class we compute the rank by the average
        # of the score
        self._score = np.sum(aux, axis=0)
        self._rank = np.argsort(self._score)[::-1]



