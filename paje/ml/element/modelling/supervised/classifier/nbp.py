from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from paje.searchspace.configspace import HPTree
from paje.ml.element.modelling.supervised.classifier.classifier import Classifier


class NBP(Classifier):
    """NB that needs positive values."""

    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.

        newconfig = self.config.copy()
        self.nb_type = newconfig.get('@nb_type')
        del newconfig['@nb_type']

        if self.nb_type == "MultinomialNB":
            self.model = MultinomialNB()
        elif self.nb_type == "ComplementNB":
            self.model = ComplementNB()
        else:
            raise Exception('Wrong NB!')

    @classmethod
    def cs_impl(self):
        node = {'@nb_type': ['c', ["MultinomialNB", "ComplementNB"]]}
        return HPTree(node=node, children=[])
