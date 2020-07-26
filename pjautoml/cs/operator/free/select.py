from pjautoml.cs.operator.nary import NAry
from pjml.util.distributions import choice


class Select(NAry):
    """TODO."""

    def sample(self):
        cs = choice(self.css)
        return cs.sample()
