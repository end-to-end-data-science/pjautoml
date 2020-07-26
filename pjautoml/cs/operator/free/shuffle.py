import numpy as np

from pjautoml.cs.operator.nary import NAry
from pjml.operator.pipeline import Pipeline


class Shuffle(NAry):
    """A permutation is sampled."""

    def sample(self):
        css = self.css.copy()
        np.random.shuffle(css)
        return Pipeline(components=[cs.sample() for cs in css], *self.kwargs)
