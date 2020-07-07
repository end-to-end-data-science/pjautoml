from pymfe.mfe import MFE as PYMFE
from paje.base.component import Component
import numpy as np

from paje.ml.element.element import Element


class MFE(Element):
    def cs_impl(self):
        raise Exception('Specify parameters like "supervised"/"unsupervised"'
                        'in the HP tree?')

    def build_impl(self):
        self.model = PYMFE()

    def apply_impl(self, data):
        return self.use_impl(data)

    def use_impl(self, data):
        self.model.fit(*data.Xy)
        names, values = self.model.extract(suppress_warnings=True)
        l = np.array(values)
        # TODO: suppressing NaNs with 0s!!
        l[~np.isfinite(l)] = 0
        return data.updated(self, l=l)
