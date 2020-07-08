from abc import abstractmethod, ABC

from pjdata.aux.serialization import materialize
from pjautoml.config.description.cs.abc.configspace import ConfigSpace
from pjautoml.config.description.distributions import choice
from pjautoml.config.description.node import Node


class ComponentCS(ConfigSpace, ABC):
    def __init__(self, name, path, config_spaces, nodes):
        if nodes is None:
            nodes = []
        jsonable = {'component': {'name': name, 'path': path}, 'nodes': nodes}
        if config_spaces:
            config_spaces = [compo.cs for compo in config_spaces]
            jsonable['components'] = config_spaces
        super().__init__(jsonable)
        for cs in nodes:
            if not isinstance(cs, Node):
                raise Exception(
                    f'{self.__class__.__name__} can only have Node as nodes.'
                    f' Not {type(cs)} !'
                )
        self._name, self.path, self.nodes = name, path, nodes
        self.config_spaces = config_spaces

    @abstractmethod
    def _sample_cfg(self):
        pass

    def sample(self):
        """Sample a completely configured component.

        Choose a path from tree and set values to parameters according to
        the given sampling functions.

        Returns
        -------
        A component
        """
        config = self._sample_cfg()

        # Fill config with values from internal nodes.
        if self.nodes:
            child_node = choice(self.nodes)
            config.update(child_node.partial_sample())

        return materialize(self.name, self.path, config)

    @abstractmethod
    def updated(self, nodes):
        pass
