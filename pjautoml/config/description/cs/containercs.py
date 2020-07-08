from pjautoml.config.description.cs.abc.componentcs import ComponentCS


class ContainerCS(ComponentCS):
    """

    Parameters
    ----------
    config_spaces
        Multiple CS.
    """

    def __init__(self, name, path, config_spaces, nodes=None):
        super().__init__(name, path, config_spaces, nodes)

    def _sample_cfg(self):
        return {'components': [c.sample() for c in self.config_spaces]}

    def identified(self, name, path):
        """Useful to fill name/path after the component has the CS built."""
        return self.__class__(name, path, self.config_spaces, *self.nodes)

    def updated(self, nodes):
        return self.__class__(self.name, self.path, self.config_spaces, nodes)
