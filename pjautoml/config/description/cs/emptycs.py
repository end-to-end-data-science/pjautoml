from pjdata.aux.serialization import materialize
from pjautoml.config.description.cs.abc.configspace import ConfigSpace


class EmptyCS(ConfigSpace):
    """CS for a component without settings, often a NoOp.

    Parameters
    ----------
    name
        Name (usually the Python class) of the component.
    path
        Path (usually the Python module) of the component.
    """

    def __init__(self, name=None, path=None):
        super().__init__({'component': {'name': name, 'path': path}})
        print("NAME: ", name)
        print("PATH: ", path)
        self.name, self.path = name, path

    def sample(self):
        return materialize(self.name, self.path, {})

    def identified(self, name, path):
        return self.__class__(name=name, path=path)
