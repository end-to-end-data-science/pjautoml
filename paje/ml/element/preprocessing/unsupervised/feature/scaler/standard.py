from sklearn.preprocessing import StandardScaler

from paje.searchspace.configspace import HPTree
from paje.ml.element.preprocessing.unsupervised.feature.scaler.scaler import Scaler

class Standard(Scaler):
    def build_impl(self):
        newconfig = self.config.copy()
        mean_std = newconfig.get('@with_mean/std')
        if mean_std is None:
            with_mean, with_std = True, True
        else:
            del newconfig['@with_mean/std']
            with_mean, with_std = mean_std
        self.model = StandardScaler(with_mean, with_std, **newconfig)

    @classmethod
    def cs_impl(cls, data=None):
        node = {
            '@with_mean/std':
                ['c', [(True, False), (False, True), (True, True)]]
            # (False, False) seems to be useless
        }
        return HPTree(node, children=[])
