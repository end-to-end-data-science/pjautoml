from abc import ABC, abstractmethod

from paje.searchspace.configspace import HPTree
from paje.ml.element.element import Element


class Reductor(Element, ABC):
    def apply_impl(self, data):
        self.att_labels = data.columns
        max_components = min(data.n_instances(), data.n_attributes())
        if hasattr(self.model, 'n_clusters'):  # DRFTAG changes terminology
            self.model.n_components = self.model.n_clusters

        # TODO: DRFTAG breaks when: Found array with 1 feature(s)
        #  (shape=(49, 1)) while a minimum of 2 is required by
        #  FeatureAgglomeration.
        # TODO: DRICA ValueError: array must not contain infs or NaNs
        self.model.fit(data.X)
        return self.use_impl(data)

    def use_impl(self, data):
        return data.updated(self, X=self.model.transform(data.X))

    @classmethod
    def cs_impl(cls, data):
        cls.check_data(data)
        # TODO: set random_state
        node = {'n_components': ['z', [1, data.n_attributes()]]}
        node.update(cls.specific_node(data))
        return HPTree(node, children=[])

    @classmethod
    @abstractmethod
    def specific_node(cls, data):
        pass
