from pjautoml.cs.cs import CS
from pjautoml.cs.operand.graph.node import Node
from pjdata.aux.serialization import materialize
from pjml.util.distributions import choice


class Graph(CS):
    """TODO."""

    def __init__(self, name=None, path=None, nodes=None):
        if nodes is None:
            raise Exception("Graph should have a list of nodes!")
        jsonable = {"component": {"name": name, "path": path}, "nodes": nodes}
        self._name, self.path, self.nodes = name, path, nodes
        self._check_all_nodes()

        super().__init__(jsonable)

    def _check_all_nodes(self):
        for nd in self.nodes:
            if not isinstance(nd, Node):
                raise Exception(
                    f"List of nodes must have only 'Node' "
                    f"objects and not '{type(nd)}' !"
                )

    def _sample_cfg(self):
        return {}

    def sample(self):
        """TODO."""
        config = self._sample_cfg()

        # Fill config with values from internal nodes.
        if self.nodes:
            child_node = choice(self.nodes)
            config.update(child_node.partial_sample())

        return materialize(self.name, self.path, config)

    def identified(self, name, path):
        return self.__class__(name, path, self.nodes)

    def updated(self, nodes):
        return self.__class__(self.name, self.path, nodes)
