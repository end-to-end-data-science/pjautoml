from pjautoml.cs.operator.nary import NAry
from pjml.operator.pipeline import Pipeline


class Chain(NAry):
    """TODO."""

    def sample(self):
        return Pipeline(components=[cs.sample() for cs in self.css], **self.kwargs)
