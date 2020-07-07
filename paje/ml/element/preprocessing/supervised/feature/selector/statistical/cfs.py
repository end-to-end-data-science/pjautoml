from paje.ml.element.preprocessing.supervised.feature.selector.filter import Filter
from skfeature.function.statistical_based import CFS
from paje.searchspace.configspace import HPTree
import pandas as pd


class FilterCFS(Filter):
    def build_impl(self):
        self.__rank = self.__score = self._selected = None
        self.model = 42 # TODO: better model here?


    def apply_impl(self, data):
        X, y = data.Xy

        # TODO: verify if is possible implement this with numpy
        y = pd.Categorical(y).codes
        self._selected = CFS.cfs(X, y)
        self._selected = self._selected[self._selected >= 0]

        # self.fit(data.X, data.Y)
        return self.use_impl(data)

    def selected(self):
        return self._selected

    def cs_impl(cls, data):
        return HPTree(node={}, children=[])
