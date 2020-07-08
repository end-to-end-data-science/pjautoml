from abc import ABC

from pjautoml.config.description.cs.abc.configspace import ConfigSpace


class OperatorCS(ConfigSpace, ABC):

    def __init__(self, *components):
        components = [compo.cs for compo in components]
        super().__init__({'components': components})
        self.components = components
