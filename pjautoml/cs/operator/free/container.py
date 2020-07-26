from pjautoml.cs.operator.nary import NAry
from pjdata.aux.serialization import materialize


class Container(NAry):
    """TODO."""

    def __init__(self, *args, seed, name, path, **kwargs):
        super().__init__(*args)
        self.name = name
        self.path = path

        self.kwargs['seed'] = seed

    def sample(self):
        """TODO."""
        config = {'components': [c.sample() for c in self.css]}
        config.update(self.kwargs)
        return materialize(self.name, self.path, config)
