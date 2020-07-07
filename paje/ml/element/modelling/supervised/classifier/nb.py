from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from paje.searchspace.hp import CatHP
from paje.searchspace.configspace import ConfigSpace
from paje.ml.element.modelling.supervised.classifier.classifier import \
    Classifier
from paje.util.distributions import choice


class NB(Classifier):
    """NB that accepts any values."""
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.
        self.nb_type = self.config['@nb_type']

        if self.nb_type == "GaussianNB":
            self.model = GaussianNB()
        elif self.nb_type == "BernoulliNB":
            self.model = BernoulliNB()
        else:
            raise Exception('Wrong NB!')

    @classmethod
    def cs_impl(cls):
        hps = {
            '@nb_type': CatHP(choice, items=['GaussianNB', 'BernoulliNB'])
        }
        return ConfigSpace(name='NB', hps=hps)
