from imblearn.under_sampling import RandomUnderSampler

from paje.searchspace.configspace import HPTree
from paje.ml.element.preprocessing.supervised.instance.sampler.resampler \
    import Resampler


class RanUnderSampler(Resampler):
    def build_impl(self):
        self.model = RandomUnderSampler(**self.config)

    @classmethod
    def cs_impl(cls, data=None):
        node = {'sampling_strategy': ['c', ['majority', 'not minority',
                                           'not majority', 'all']]}
        return HPTree(node, children=[])
