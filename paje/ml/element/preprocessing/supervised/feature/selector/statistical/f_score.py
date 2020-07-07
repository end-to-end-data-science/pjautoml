from paje.ml.element.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import f_score
import pandas as pd
import math


class FilterFScore(Filter):
    """  """
    def apply_impl(self, data):
        X, y = data.Xy

        # TODO: verify if it is possible implement this with numpy
        y = pd.Categorical(y).codes

        self._score = f_score.f_score(X, y)
        self._rank = f_score.feature_ranking(self._score)
        self._nro_features = math.ceil(self.ratio * X.shape[1])

        return self.use_impl(data)
