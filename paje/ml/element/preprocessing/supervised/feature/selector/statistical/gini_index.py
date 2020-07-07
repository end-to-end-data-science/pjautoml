import pandas as pd

from paje.ml.element.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import gini_index
import math

class FilterGiniIndex(Filter):
    """  """
    def apply_impl(self, data):
        # TODO: verify if is possible implement this with numpy
        X, y = data.Xy
        y = pd.Categorical(y).codes

        self._score = gini_index.gini_index(X, y)
        self._rank = gini_index.feature_ranking(self._score)
        self._nro_features = math.ceil(self.ratio * X.shape[1])

        return self.use_impl(data)
