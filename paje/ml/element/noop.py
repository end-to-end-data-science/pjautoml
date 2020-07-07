from paje.searchspace.configspace import HPTree
from paje.ml.element.element import Element


class Noop(Element):
    def build_impl(self):
        self.model = 42 # TODO: better model here?

    def apply_impl(self, data):
        return data

    def use_impl(self, data):
        return data

    def cs_impl(cls, data):
        HPTree({}, [])
